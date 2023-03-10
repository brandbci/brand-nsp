#!/usr/bin/env python
# -*- coding: utf-8 -*-
# threshold_extraction.py
# Kevin Bodkin, Yahia Ali

from copy import deepcopy
import gc
import json
import logging
import signal
import sys
import time
import yaml

import numpy as np
import scipy.signal
from brand import BRANDNode
from brand.redis import xread_count


class ThresholdExtraction(BRANDNode):

    def __init__(self):
        super().__init__()

        # threshold multiplier, usually around -5
        self.thresh_mult = self.parameters['thresh_mult']
        # number of Redis packets to read on each iteration
        self.pack_per_call = self.parameters['pack_per_call']
        # whether to export the filtered data
        self.output_filtered = self.parameters['output_filtered']
        # enable acausal filtering
        self.acausal_filter = self.parameters['acausal_filter']
        self.causal = not self.acausal_filter
        # length of the buffer used for acausal filtering
        self.acausal_filter_lag = self.parameters['acausal_filter_lag']
        # whether to conduct common-average referencing
        self.demean = self.parameters['enable_CAR']

        # parameters of the input stream
        self.input_stream = self.parameters['input_name']
        # number of samples per channel per redis entry
        self.samp_per_stream = self.parameters['input_samp_per_stream']
        # number of channels
        self.n_channels = self.parameters['input_chan_per_stream']

        # whether to remove coincident spikes
        self.num_coincident = self.parameters['num_coincident_spikes'] if 'num_coincident_spikes' in self.parameters else None
        self.num_coincident = self.n_channels+1 if self.num_coincident is None else self.num_coincident

        # thresholds stream
        if 'thresholds_stream' in self.parameters:
            self.thresholds_stream = self.parameters['thresholds_stream']
        else:
            self.thresholds_stream = None

        # thresholds file
        if 'thresholds_file' in self.parameters:
            self.thresholds_file = self.parameters['thresholds_file']
        else:
            self.thresholds_file = None

        # which thresholds to use
        if (self.thresholds_stream is not None
                or self.thresholds_file is not None):
            if 'thresholds_ch_range' in self.parameters:
                if len(self.parameters['thresholds_ch_range']) == 2:
                    self.th_chans = range(
                        self.parameters['thresholds_ch_range'][0],
                        self.parameters['thresholds_ch_range'][1])
                else:
                    logging.warning(
                        '\'thresholds_ch_range\' parameter should be length 2,'
                        ' attempting to use all channels in the thresholds '
                        'stream or file')
                    self.th_chans = None
            else:
                logging.warning(
                    '\'thresholds_ch_range\' was not provided, attempting to '
                    'use all channels in the thresholds stream or file')
                self.th_chans = None

        # optional datatype
        if 'input_data_type' in self.parameters:
            self.dtype = self.parameters['input_data_type']
        else:
            self.dtype = np.int16

        # list of lists of common-average reference groupings
        if self.demean and 'CAR_group_sizes' in self.parameters:
            car_sizes = self.parameters['CAR_group_sizes']
            if not isinstance(car_sizes, list):
                if isinstance(car_sizes, int):
                    car_sizes = []
                    ch_count = deepcopy(self.n_channels)

                    # get CAR group sizes of the specified size, until we run
                    # out of channels for the stream
                    while ch_count > 0:
                        car_sizes.append(
                            min([self.parameters['CAR_group_sizes'],
                                 ch_count]))
                        ch_count -= self.parameters['CAR_group_sizes']

            self.car_groups = []
            ch_count = 0
            for g in car_sizes:
                if not isinstance(g, int):
                    raise ValueError(
                        '\'CAR_group_sizes\' must be a list of \'int\'s or a '
                        'single \'int\', but '
                        f'{self.parameters["CAR_group_sizes"]} was given')
                self.car_groups.append(
                    np.arange(ch_count, ch_count + g).tolist())
                ch_count += g
        else:
            self.car_groups = [np.arange(0, self.n_channels).tolist()]

        # exclude channels
        if 'exclude_channels' in self.parameters:
            exclude_ch = self.parameters['exclude_channels']
            if not isinstance(exclude_ch, list):
                if isinstance(exclude_ch, int):
                    exclude_ch = [exclude_ch]
            for c in exclude_ch:
                if not isinstance(c, int):
                    raise ValueError(
                        '\'exclude_channels\' must be a list of \'int\'s or'
                        ' a single \'int\', but '
                        f'{self.parameters["exclude_channels"]} was given.'
                        ' Exiting')
            for c in exclude_ch:
                for g in self.car_groups:
                    if c in g:
                        g.remove(c)

        # stream containing the list of channels to use
        if 'ch_mask_stream' in self.parameters:
            ch_mask_entry = self.r.xrevrange(self.parameters['ch_mask_stream'], '+', '-', count=1)
            if ch_mask_entry:
                self.ch_mask = np.frombuffer(ch_mask_entry[0][1][b'channels'], dtype=np.uint16)
                if self.th_chans is None:
                    logging.warning(f'\'ch_mask_stream\' was provided but \'thresholds_ch_range\' was not, so the incorrect channels may be masked')
                else:
                    self.ch_mask = np.array(list(set(self.ch_mask).intersection(set(self.th_chans))))
                    self.ch_mask -= self.th_chans[0]
            else:
                logging.warning(f'\'ch_mask_stream\' was set to {self.parameters["ch_mask_stream"]}, but there were no entries. Defaulting to using all channels')
                self.ch_mask = np.arange(self.n_channels, dtype=np.uint16)
        else:
            self.ch_mask = np.arange(self.n_channels, dtype=np.uint16)

        # keep only masked channels
        for g_idx in range(len(self.car_groups)):
            self.car_groups[g_idx] = list(set(self.car_groups[g_idx]).intersection(set(self.ch_mask)))

        # define timing and sync keys
        if 'sync_key' in self.parameters:
            self.sync_key = self.parameters['sync_key'].encode()
        else:
            self.sync_key = b'sync'
        if 'time_key' in self.parameters:
            self.time_key = self.parameters['time_key'].encode()
        else:
            self.time_key = b'ts'
        if 'sync_source_id' in self.parameters:
            self.sync_source_id = self.parameters['sync_source_id']
        else:
            self.sync_source_id = 'i'

        # build filtering pipeline
        if self.causal:
            self.filter_func, self.sos, self.zi = self.build_filter()
        else:
            (self.filter_func, self.sos, self.zi, self.rev_win,
             self.rev_zi) = self.build_filter()

        # load or compute thresholds
        self.thresholds = None
        if self.thresholds_stream is not None:
            self.thresholds = self.load_thresholds_from_stream(
                self.thresholds_stream, self.th_chans)

        if self.thresholds is None and self.thresholds_file is not None:
            self.thresholds = self.load_thresholds_from_file(
                self.thresholds_file, self.th_chans)

        if self.thresholds is None:
            # amount of data to use for threshold calculation
            self.thresh_calc_len = self.parameters['thresh_calc_len']
            self.thresholds = self.calc_thresh(
                self.input_stream, self.thresh_mult, self.thresh_calc_len,
                self.samp_per_stream, self.n_channels, self.sos, self.zi)

        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)

    def build_filter(self):
        # order of the butterworth filter
        but_order = self.parameters['butter_order']
        # lower cutoff frequency
        but_low = self.parameters['butter_lowercut']
        # upper cutoff frequency
        but_high = self.parameters['butter_uppercut']
        # whether to use an FIR filter on the reverse-pass
        acausal_filter = self.acausal_filter
        # enable causal filtering
        causal = self.causal
        # sampling rate of the input
        fs = self.parameters['input_samp_freq']

        # determine filter type
        if but_low and but_high:
            filt_type = 'bandpass'
            Wn = [but_low, but_high]
        elif but_high:
            filt_type = 'lowpass'
            Wn = but_high
        elif but_low:
            filt_type = 'highpass'
            Wn = but_low
        else:
            raise ValueError(
                "Must specify 'butter_lowercut' or 'butter_uppercut'")

        # set up filter
        sos = scipy.signal.butter(but_order,
                                  Wn,
                                  btype=filt_type,
                                  analog=False,
                                  output='sos',
                                  fs=fs)  # set up a filter
        # initialize the state of the filter
        zi_flat = scipy.signal.sosfilt_zi(sos)
        # so that we have the right number of dimensions
        zi = np.zeros((zi_flat.shape[0], self.n_channels, zi_flat.shape[1]))
        # filter initialization
        for ii in range(self.n_channels):
            zi[:, ii, :] = zi_flat

        # select the filtering function
        if acausal_filter and acausal_filter.lower() == 'fir':
            use_fir = True
        else:
            use_fir = False
        filter_func = get_filter_func(self.demean, causal, use_fir=use_fir)

        # log the filter info
        causal_str = 'causal' if causal else 'acausal'
        message = (f'Loading {but_order :d} order, '
                   f'{Wn} hz {filt_type} {causal_str}')
        if causal:
            message += ' IIR filter'
        elif use_fir:
            message += ' IIR-FIR filter'
        else:
            message += ' IIR-IIR filter'
        message += ' with CAR' if self.demean else ''
        logging.info(message)

        if not causal:
            if use_fir:
                # FIR filter (backward)
                N = self.acausal_filter_lag + 1  # length of the filter
                imp = scipy.signal.unit_impulse(N)
                rev_win = scipy.signal.sosfilt(sos, imp)
                # filter initialization
                rev_zi_flat = scipy.signal.lfilter_zi(rev_win, 1.0)
                rev_zi = np.zeros((self.n_channels, rev_zi_flat.shape[0]))
                for ii in range(self.n_channels):
                    rev_zi[ii, :] = rev_zi_flat
            else:
                rev_win = None
                rev_zi = zi.copy()

        if causal:
            return filter_func, sos, zi
        else:
            return filter_func, sos, zi, rev_win, rev_zi

    def calc_thresh(self, stream, thresh_mult, thresh_cal_len, samp_per_stream,
                    n_channels, sos, zi):
        reply = xread_count(self.r,
                            stream=stream,
                            startid='$',
                            count=thresh_cal_len,
                            block=0)

        _, entries = reply[0]  # get the list of entries

        read_arr = np.empty((n_channels, thresh_cal_len * samp_per_stream),
                            dtype=np.float32)
        filt_arr = np.empty((n_channels, thresh_cal_len * samp_per_stream),
                            dtype=np.float32)
        read_times = np.empty((thresh_cal_len * samp_per_stream))

        i_start = 0
        for _, entry_data in entries:  # put it all into an array
            i_end = i_start + samp_per_stream
            read_arr[:, i_start:i_end] = np.reshape(
                np.frombuffer(entry_data[b'samples'], dtype=self.dtype),
                (n_channels, samp_per_stream))
            read_times[i_start:i_end] = np.frombuffer(
                entry_data[b'timestamps'], np.uint32)
            i_start = i_end

        if self.causal:
            self.filter_func(read_arr, filt_arr, sos, zi)
        else:
            if self.demean:
                common_average_reference(read_arr, self.car_groups)
            filt_arr[:, :] = scipy.signal.sosfiltfilt(sos, read_arr, axis=1)

        thresholds = (thresh_mult *
                      np.sqrt(np.mean(np.square(filt_arr), axis=1))).reshape(
                          -1, 1)

        # log thresholds to database
        thresolds_enc = thresholds.astype(np.int16).tobytes()
        self.r.xadd(f'{self.NAME}_thresholds', {b'thresholds': thresolds_enc})

        logging.info('Calculated and set thresholds')
        return thresholds

    def load_thresholds_from_file(self, thresholds_file, tf_chans):
        try:
            with open(thresholds_file, 'r') as f:
                thresh_yaml = yaml.safe_load(f)
            if 'thresholds' in thresh_yaml:
                if tf_chans is None:
                    if len(thresh_yaml['thresholds']) == self.n_channels:
                        logging.info(
                            f'Loaded thresholds from {thresholds_file}')
                        return np.array(thresh_yaml['thresholds']).reshape(
                            -1, 1)
                    else:
                        raise ValueError(
                            f'Number of thresholds in {thresholds_file} '
                            f'({len(thresh_yaml["thresholds"])}) does not '
                            f'equal n_channels parameter {(self.n_channels)}')
                # if all of our requested channels are in the available range
                # of channels
                elif (set(tf_chans)
                      & set(range(0, len(
                          thresh_yaml['thresholds'])))) == set(tf_chans):
                    logging.info(f'Loaded thresholds from {thresholds_file}')
                    return np.array(
                        thresh_yaml['thresholds'])[tf_chans].reshape(-1, 1)
                else:
                    raise ValueError(
                        f'Channel range '
                        f'{self.parameters["thresholds_ch_range"]} outside of '
                        f'available channels in {thresholds_file} '
                        f'(max {len(thresh_yaml["thresholds"])})')
            else:
                logging.warning(
                    f'Could not find \'thresholds\' key in {thresholds_file}')
                return None

        except FileNotFoundError:
            logging.warning(
                f'Could not find thresholds file at {thresholds_file}')
            return None

    def load_thresholds_from_stream(self, stream, th_chans):
        entry = self.r.xrevrange(stream, '+', '-', count=1)
        if entry:
            thresholds = np.frombuffer(entry[0][1][b'thresholds'],
                                       dtype=np.float64)
            if th_chans is None:
                if len(thresholds) == self.n_channels:
                    logging.info(f'Loaded thresholds from the {stream} stream')
                    return thresholds.reshape(-1, 1)
                else:
                    raise ValueError(
                        f'Number of thresholds in the {stream} stream '
                        f'({len(thresholds)}) does not equal n_channels '
                        f'parameter {(self.n_channels)}')
            # if all of our requested channels are in the available range of
            # channels
            elif (set(th_chans)
                  & set(range(0, len(thresholds)))) == set(th_chans):
                logging.info(f'Loaded thresholds from the {stream} stream')
                return thresholds[th_chans].reshape(-1, 1)
            else:
                raise ValueError(
                    f'Channel range {self.parameters["thresholds_ch_range"]} '
                    f'outside of available channels in {stream} stream '
                    f'(max {len(thresholds)})')
        else:
            logging.warning(f'{stream} stream has no entries')
            return None

    def run(self):
        # get class variables
        if not self.causal:
            rev_win = self.rev_win
            rev_zi = self.rev_zi
        input_stream = self.input_stream
        output_filtered = self.output_filtered
        pack_per_call = self.pack_per_call
        samp_per_stream = self.samp_per_stream
        sos = self.sos
        thresholds = self.thresholds
        zi = self.zi

        # initialize arrays
        n_samp = samp_per_stream * pack_per_call
        data_buffer = np.zeros((self.n_channels, n_samp), dtype=np.float32)
        filt_buffer = np.zeros_like(data_buffer)
        rev_buffer = np.zeros(
            (self.n_channels, self.acausal_filter_lag + n_samp),
            dtype=np.float32)
        crossings = np.zeros_like(data_buffer)
        samp_times = np.zeros(n_samp, dtype=np.uint32)
        buffer_len = rev_buffer.shape[1]
        samp_times_buffer = np.zeros(buffer_len, dtype=np.uint32)
        buffer_fill = 0  # how many samples have been read into the buffer

        # initialize stream entries
        cross_dict = {}
        filt_dict = {}
        coinc_dict = {}
        sync_dict = {self.sync_source_id: int(samp_times[0])}

        # initialize xread stream dictionary
        input_stream_dict = {input_stream: '$'}

        # set timeout
        timeout = 500

        # name the filtered output stream
        filt_stream = f'{self.NAME}_filt'
        while True:
            # wait to get data from cerebus stream, then parse it
            xread_receive = self.r.xread(input_stream_dict,
                                         block=timeout,
                                         count=pack_per_call)

            # only run this if we have data
            if len(xread_receive) >= pack_per_call:
                indStart = 0
                # run each entry individually
                for entry_id, entry_data in xread_receive[0][1]:
                    indEnd = indStart + samp_per_stream
                    data_buffer[:, indStart:indEnd] = np.reshape(
                        np.frombuffer(entry_data[b'samples'],
                                      dtype=self.dtype),
                        (self.n_channels, samp_per_stream))
                    samp_times[indStart:indEnd] = np.frombuffer(
                        entry_data[b'timestamps'], np.uint32)
                    indStart = indEnd

                # update key to be the entry number of last item in list
                input_stream_dict[input_stream] = entry_id

                # filter the data and find threshold times
                if self.causal:
                    self.filter_func(data_buffer, filt_buffer, sos, zi,
                                     self.car_groups)
                else:
                    self.filter_func(data_buffer,
                                     filt_buffer,
                                     rev_buffer,
                                     sos=sos,
                                     zi=zi,
                                     group_list=self.car_groups,
                                     rev_win=rev_win,
                                     rev_zi=rev_zi)
                    # update sample time buffer
                    samp_times_buffer[:-n_samp] = (samp_times_buffer[n_samp:])
                    samp_times_buffer[-n_samp:] = samp_times

                # find for each channel along the first dimension, keep dims,
                # pack into a byte object and put into the thresh crossings
                # dict
                if self.causal:
                    samp_time_current = samp_times[:n_samp]
                else:
                    samp_time_current = samp_times_buffer[:n_samp]
                    # check the buffer
                    if buffer_fill + n_samp < buffer_len:  # buffer is not full
                        buffer_fill += n_samp  # count the samples added
                        continue  # skip writing to Redis

                sync_dict[self.sync_source_id] = int(samp_time_current[0])
                cross_dict[self.sync_key] = json.dumps(sync_dict)
                cross_dict[b'timestamps'] = samp_time_current[0].tobytes()
                # is there a threshold crossing in the last ms?
                crossings[:, 1:] = ((filt_buffer[:, 1:] < thresholds) &
                                    (filt_buffer[:, :-1] >= thresholds))
                cross_now = np.any(crossings, axis=1).astype(np.int16)

                # Redis
                p = self.r.pipeline()  # create a new pipeline

                time_now = np.uint64(time.monotonic_ns()).tobytes()

                # coincident spike removal
                tot_spikes = cross_now.sum()
                if tot_spikes >= self.num_coincident:
                    logging.info(f'{tot_spikes} coincident spikes detected, timestamp: {int(samp_time_current[0])}')
                    coinc_dict[self.sync_key] = json.dumps(sync_dict)
                    coinc_dict[b'timestamps'] = samp_time_current[0].tobytes()
                    coinc_dict[self.time_key] = time_now
                    coinc_dict[b'crossings'] = cross_now.tobytes()
                    p.xadd(f'{self.NAME}_coinc', coinc_dict)
                    cross_now[:] = 0

                    
                cross_dict[b'crossings'] = cross_now.tobytes()

                # log timestamps
                cross_dict[self.time_key] = time_now

                # thresholdCrossings stream
                p.xadd(self.NAME, cross_dict)
                # filtered data stream
                if output_filtered:
                    # if we're storing the filtered data
                    filt_dict[self.sync_key] = json.dumps(sync_dict)
                    filt_dict[b'timestamps'] = samp_time_current.tobytes()
                    filt_dict[b'samples'] = filt_buffer.astype(
                        np.int16).tobytes()
                    filt_dict[self.time_key] = time_now
                    # add the filtered stuff to the pipeline
                    p.xadd(filt_stream, filt_dict)

                # write to Redis
                p.execute()

            elif len(xread_receive) == 0:
                logging.warning("No neural data has been received in the"
                                f" last {timeout} ms")

    def terminate(self, *_):
        logging.info('SIGINT received, Exiting')
        gc.collect()
        sys.exit(0)


# Filtering functions
def get_filter_func(demean, causal=False, use_fir=True):
    """
    Get a function for filtering the data

    Parameters
    ----------
    demean : bool
        Whether to apply a common average reference before filtering
    causal : bool
        Whether to use causal filtering or acausal filtering
    use_fir : bool
        Whether to use an FIR filter for the reverse filter (when causal=False)
    """

    def causal_filter(data, filt_data, sos, zi, group_list):
        """
        causal filtering

        Parameters
        ----------
        data : array_like
            An N-dimensional input array.
        filt_data : ndarray
            Array to store the output of the digital filter.
        sos : array_like
            Array of second-order filter coefficients
        zi : array_like
            Initial conditions for the cascaded filter delays
        group_list : list
            List of lists of channels grouped together across
            which to compute a common average reference
        """
        if demean:
            common_average_reference(data, group_list)
        filt_data[:, :], zi[:, :] = scipy.signal.sosfilt(sos,
                                                         data,
                                                         axis=1,
                                                         zi=zi)

    def acausal_filter(data,
                       filt_data,
                       rev_buffer,
                       sos,
                       zi,
                       group_list,
                       rev_win=None,
                       rev_zi=None):
        """
        acausal filtering

        Parameters
        ----------
        data : array_like
            An N-dimensional input array.
        filt_data : ndarray
            Array to store the output of the digital filter.
        rev_buffer : ndarray
            Array to store the output of the forward IIR filter.
        sos : array_like
            Array of second-order filter coefficients
        zi : array_like
            Initial conditions for the cascaded filter delays
        group_list : list
            List of lists of channels grouped together across
            which to compute a common average reference
        rev_win : array-like, optional
            Coefficients of the reverse FIR filter
        rev_zi : array-like, optional
            Steady-state conditions of the reverse filter
        """
        if demean:
            common_average_reference(data, group_list)

        # shift the buffer
        n_samp = data.shape[1]
        rev_buffer[:, :-n_samp] = rev_buffer[:, n_samp:]

        # run the forward pass filter
        rev_buffer[:, -n_samp:], zi[:, :] = scipy.signal.sosfilt(sos,
                                                                 data,
                                                                 axis=1,
                                                                 zi=zi)
        # run the backward pass filter
        # 1. pass in the reversed buffer
        # 2. get the last N samples of the filter output
        # 3. reverse the output when saving to filt_data
        if use_fir:
            for ii in range(filt_data.shape[0]):
                filt_data[ii, ::-1] = np.convolve(rev_buffer[ii, ::-1],
                                                  rev_win, 'valid')
        else:
            ic = rev_zi * filt_data[:, -1][None, :, None]
            filt_data[:, ::-1] = scipy.signal.sosfilt(sos,
                                                      rev_buffer[:, ::-1],
                                                      axis=1,
                                                      zi=ic)[0][:, -n_samp:]

    filter_func = causal_filter if causal else acausal_filter

    return filter_func


def common_average_reference(data, group_list):
    """
    common average reference by group

    Parameters
    ----------
    data : array_like
        An 2-dimensional input array with shape
        [channel x time]
    group_list : list
        List of lists of channels grouped together across
        which to compute a common average reference
    """
    for g in group_list:
        data[g, :] -= data[g, :].mean(axis=0, keepdims=True)


if __name__ == '__main__':
    gc.disable()  # disable garbage collection

    thresh_ext = ThresholdExtraction()
    thresh_ext.run()

    gc.collect()  # run garbage collection
