#!/usr/bin/env python
# -*- coding: utf-8 -*-
# bpExtraction.py
# Yahia Ali
import gc
import json
import logging
import signal
import time
from copy import deepcopy

import numpy as np
import scipy.signal

from brand import BRANDNode


class bpExtraction(BRANDNode):

    def __init__(self):
        super().__init__()
        # number of Redis packets to read on each iteration
        self.pack_per_call = self.parameters['pack_per_call']
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

        # optional datatype
        if 'input_data_type' in self.parameters:
            self.dtype = self.parameters['input_data_type']
        else:
            self.dtype = np.int16

        # list of lists of common-average reference groupings
        if self.demean and 'CAR_group_sizes' in self.parameters:
            car_sizes = self.parameters['CAR_group_sizes']
            if not isinstance(car_sizes, list) and isinstance(car_sizes, int):
                car_size_list = []
                ch_count = deepcopy(self.n_channels)
                # get CAR group sizes of the specified size, until we run
                # out of channels for the stream
                while ch_count > 0:
                    car_size_list.append(min([car_sizes, ch_count]))
                    ch_count -= car_sizes
                car_sizes = car_size_list

            self.car_groups = []
            ch_count = 0
            for g in car_sizes:
                if not isinstance(g, int):
                    raise ValueError(
                        "'CAR_group_sizes' must be a list of 'int's or a "
                        "single 'int', but "
                        f"{self.parameters['CAR_group_sizes']} was given")
                self.car_groups.append(np.arange(ch_count, ch_count + g))
                ch_count += g
        else:
            self.car_groups = [np.arange(self.n_channels)]

        # exclude channels
        if 'exclude_channels' in self.parameters:
            exclude_ch = self.parameters['exclude_channels']
            if not isinstance(exclude_ch, list) and isinstance(
                    exclude_ch, int):
                exclude_ch = [exclude_ch]
            for c in exclude_ch:
                if not isinstance(c, int):
                    raise ValueError(
                        "'exclude_channels' must be a list of 'int's or"
                        " a single 'int', but "
                        f"{self.parameters['exclude_channels']} was given."
                        " Exiting")
            for i, g in enumerate(self.car_groups):
                self.car_groups[i] = np.setdiff1d(g, exclude_ch)

        # stream containing the list of channels to use
        if 'ch_mask_stream' in self.parameters:
            ch_mask_entry = self.r.xrevrange(self.parameters['ch_mask_stream'],
                                             '+',
                                             '-',
                                             count=1)
            if ch_mask_entry:
                self.ch_mask = np.frombuffer(ch_mask_entry[0][1][b'channels'],
                                             dtype=np.uint16)
            else:
                logging.warning(
                    f'\'ch_mask_stream\' was set to '
                    f'{self.parameters["ch_mask_stream"]}, but there were no'
                    ' entries. Defaulting to using all channels')
                self.ch_mask = np.arange(self.n_channels, dtype=np.uint16)
        else:
            self.ch_mask = np.arange(self.n_channels, dtype=np.uint16)

        # keep only masked channels
        for g_idx in range(len(self.car_groups)):
            self.car_groups[g_idx] = np.intersect1d(self.car_groups[g_idx],
                                                    self.ch_mask)

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

    def run(self):
        # get class variables
        if not self.causal:
            rev_win = self.rev_win
            rev_zi = self.rev_zi
        input_stream = self.input_stream
        pack_per_call = self.pack_per_call
        samp_per_stream = self.samp_per_stream
        sos = self.sos
        zi = self.zi

        # initialize arrays
        n_samp = samp_per_stream * pack_per_call
        data_buffer = np.zeros((self.n_channels, n_samp), dtype=np.float32)
        filt_buffer = np.zeros_like(data_buffer)
        power_buffer = np.zeros(self.n_channels, dtype=np.float32)
        rev_buffer = np.zeros(
            (self.n_channels, self.acausal_filter_lag + n_samp),
            dtype=np.float32)
        samp_times = np.zeros(n_samp, dtype=np.uint32)
        buffer_len = rev_buffer.shape[1]
        samp_times_buffer = np.zeros(buffer_len, dtype=np.uint32)
        buffer_fill = 0  # how many samples have been read into the buffer

        # initialize stream entries
        power_dict = {}
        sync_dict = {self.sync_source_id: int(samp_times[0])}

        # initialize xread stream dictionary
        input_stream_dict = {input_stream: '$'}

        # set timeout
        timeout = 500

        # name the filtered output stream
        power_stream = self.NAME
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

                # filter the data
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
                # convert to power
                power_buffer[:] = np.square(filt_buffer).mean(axis=1)

                # set sync index
                if self.causal:
                    samp_time_current = samp_times[:n_samp]
                else:
                    samp_time_current = samp_times_buffer[:n_samp]
                    # check the buffer
                    if buffer_fill + n_samp < buffer_len:  # buffer is not full
                        buffer_fill += n_samp  # count the samples added
                        continue  # skip writing to Redis

                sync_dict[self.sync_source_id] = int(samp_time_current[0])

                # Redis
                p = self.r.pipeline()  # create a new pipeline

                # log timestamps
                time_now = np.uint64(time.monotonic_ns()).tobytes()

                # filtered data stream
                # if we're storing the filtered data
                power_dict[self.sync_key] = json.dumps(sync_dict)
                power_dict[b'timestamps'] = samp_time_current[0].tobytes()
                power_dict[b'samples'] = power_buffer.tobytes()
                power_dict[self.time_key] = time_now
                # add the filtered stuff to the pipeline
                p.xadd(power_stream, power_dict)

                # write to Redis
                p.execute()

            elif len(xread_receive) == 0:
                logging.warning("No neural data has been received in the"
                                f" last {timeout} ms")


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

    lfp_ext = bpExtraction()
    lfp_ext.run()

    gc.collect()  # run garbage collection
