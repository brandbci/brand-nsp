#!/usr/bin/env python
# -*- coding: utf-8 -*-
# threshold_extraction.py
# Kevin Bodkin, Yahia Ali

import gc
import json
import logging
import signal
import sys
import time

import numpy as np
import scipy.signal
from brand import BRANDNode
from brand.redis import xread_count


class ThresholdExtraction(BRANDNode):

    def __init__(self):
        super().__init__()

        # threshold multiplier, usually around -5
        self.thresh_mult = self.parameters['thresh_mult']
        # amount of data to use for threshold calculation
        self.thresh_calc_len = self.parameters['thresh_calc_len']
        # number of Redis packets to read on each iteration
        self.pack_per_call = self.parameters['pack_per_call']
        # whether to export the filtered data
        self.output_filtered = self.parameters['output_filtered']
        # enable acausal filtering
        self.acausal_filter = self.parameters['acausal_filter']
        self.causal = not self.acausal_filter
        # length of the buffer used for acausal filtering
        self.acausal_buffer_len = self.parameters['acausal_buffer_len']

        # parameters of the input stream
        self.input_params = self.parameters['input_stream']
        self.input_stream = self.input_params['name']
        # number of samples per channel per redis entry
        self.samp_per_stream = self.input_params['samp_per_stream']
        # number of channels
        self.n_channels = self.input_params['chan_per_stream']

        # build filtering pipeline
        if self.causal:
            self.filter_func, self.sos, self.zi = self.build_filter()
        else:
            (self.filter_func, self.sos, self.zi, self.rev_win,
             self.rev_zi) = self.build_filter()

        # calculate spike thresholds from the start of the data
        self.thresholds = self.calc_thresh(self.input_stream, self.thresh_mult,
                                           self.thresh_calc_len,
                                           self.samp_per_stream,
                                           self.n_channels, self.sos, self.zi)
        # log thresholds to database
        thresolds_enc = self.thresholds.astype(np.int16).tobytes()
        self.r.xadd('thresholds', {b'thresholds': thresolds_enc})

        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)

    def build_filter(self):
        # order of the butterworth filter
        but_order = self.parameters['butter_order']
        # lower cutoff frequency
        but_low = self.parameters['butter_lowercut']
        # upper cutoff frequency
        but_high = self.parameters['butter_uppercut']
        # enable common average reference
        demean = self.parameters['enable_CAR']
        # whether to use an FIR filter on the reverse-pass
        acausal_filter = self.acausal_filter
        # enable causal filtering
        causal = self.causal
        # sampling rate of the input
        fs = self.parameters['input_stream']['samp_freq']

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
        nyq = .5 * fs
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
        filter_func = get_filter_func(demean, causal, use_fir=use_fir)

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
        message += ' with CAR' if demean else ''
        logging.info(message)

        if not causal:
            if use_fir:
                # FIR filter (backward)
                N = self.acausal_buffer_len  # length of the buffer
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
                np.frombuffer(entry_data[b'samples'], np.int16),
                (n_channels, samp_per_stream))
            read_times[i_start:i_end] = np.frombuffer(
                entry_data[b'timestamps'], np.uint32)
            i_start = i_end

        if self.causal:
            self.filter_func(read_arr, filt_arr, sos, zi)
        else:
            filt_arr[:, :] = scipy.signal.sosfiltfilt(sos, read_arr, axis=1)

        thresholds = (thresh_mult *
                      np.sqrt(np.mean(np.square(filt_arr), axis=1))).reshape(
                          -1, 1)
        logging.info('Thresholds are set')
        return thresholds

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
        data_buffer = np.zeros(
            (self.n_channels, samp_per_stream * pack_per_call),
            dtype=np.float32)
        filt_buffer = np.zeros_like(data_buffer)
        fir_buffer = np.zeros((self.n_channels, self.acausal_buffer_len),
                              dtype=np.float32)
        crossings = np.zeros_like(data_buffer)
        samp_times = np.zeros(samp_per_stream * pack_per_call, dtype=np.uint32)
        samp_times_buffer = np.zeros(self.acausal_buffer_len, dtype=np.uint32)

        # initialize stream entries
        cross_dict = {}
        filt_dict = {}
        sync_dict = {'nsp_idx': int(samp_times[0])}

        # initialize xread stream dictionary
        input_stream_dict = {input_stream: '$'}

        # set timeout
        timeout = 500

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
                        np.frombuffer(entry_data[b'samples'], np.int16),
                        (self.n_channels, samp_per_stream))
                    samp_times[indStart:indEnd] = np.frombuffer(
                        entry_data[b'timestamps'], np.uint32)
                    indStart = indEnd

                # filter the data and find threshold times
                if self.causal:
                    self.filter_func(data_buffer, filt_buffer, sos, zi)
                else:
                    self.filter_func(data_buffer,
                                     filt_buffer,
                                     fir_buffer,
                                     sos=sos,
                                     zi=zi,
                                     rev_win=rev_win,
                                     rev_zi=rev_zi)
                    # update sample time buffer
                    samp_times_buffer[:-indEnd] = samp_times_buffer[indEnd:]
                    samp_times_buffer[-indEnd:] = samp_times

                # find for each channel along the first dimension, keep dims,
                # pack into a byte object and put into the thresh crossings
                # dict
                if self.causal:
                    samp_time_current = samp_times[0]
                else:
                    samp_time_current = samp_times_buffer[0]

                sync_dict['nsp_idx'] = int(samp_time_current)
                cross_dict[b'sync'] = json.dumps(sync_dict)
                cross_dict[b'timestamps'] = samp_time_current.tobytes()
                # is there a threshold crossing in the last ms?
                crossings[:, 1:] = ((filt_buffer[:, 1:] < thresholds) &
                                    (filt_buffer[:, :-1] >= thresholds))
                cross_dict[b'crossings'] = np.any(crossings, axis=1).astype(
                    np.int16).tobytes()

                # Redis
                p = self.r.pipeline()  # create a new pipeline

                # log timestamps
                time_now = np.uint64(time.monotonic_ns()).tobytes()
                cross_dict[b'BRANDS_time'] = time_now

                # thresholdCrossings stream
                p.xadd('thresholdCrossings', cross_dict)
                # filtered data stream
                if output_filtered:
                    # if we're storing the filtered data
                    filt_dict[b'timestamps'] = samp_times.tobytes()
                    filt_dict[b'samples'] = filt_buffer.astype(
                        np.int16).tobytes()
                    filt_dict[b'BRANDS_time'] = time_now
                    # add the filtered stuff to the pipeline
                    p.xadd('filteredCerebusAdapter', filt_dict)

                # write to Redis
                p.execute()

                # update key to be the entry number of last item in list
                input_stream_dict[input_stream] = entry_id
            else:
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

    def causal_filter(data, filt_data, sos, zi):
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
        """
        if demean:
            data[:, :] = data - data.mean(axis=0, keepdims=True)
        filt_data[:, :], zi[:, :] = scipy.signal.sosfilt(sos,
                                                         data,
                                                         axis=1,
                                                         zi=zi)

    def acausal_filter(data,
                       filt_data,
                       fir_buffer,
                       sos,
                       zi,
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
        fir_buffer : ndarray
            Array to store the output of the forward IIR filter.
        sos : array_like
            Array of second-order filter coefficients
        zi : array_like
            Initial conditions for the cascaded filter delays
        rev_win : array-like
            Coefficients of FIR filter
        rev_zi : array-like
            Steady-state conditions of the FIR filter
        """
        if demean:
            data[:, :] = data - data.mean(axis=0, keepdims=True)

        # shift the buffer
        n_samples = data.shape[1]
        fir_buffer[:, :-n_samples] = fir_buffer[:, n_samples:]

        # run the forward pass filter
        fir_buffer[:, -n_samples:], zi[:, :] = scipy.signal.sosfilt(sos,
                                                                    data,
                                                                    axis=1,
                                                                    zi=zi)
        # run the backward pass filter
        # 1. pass in the reversed buffer
        # 2. get the last N samples of the filter output
        # 3. reverse the output when saving to filt_data
        if use_fir:
            ic = rev_zi * filt_data[:, -1][:, None]
            filt_data[:, ::-1] = scipy.signal.lfilter(rev_win,
                                                      1.0,
                                                      fir_buffer[:, ::-1],
                                                      zi=ic)[0][:, -n_samples:]
        else:
            ic = rev_zi * filt_data[:, -1][None, :, None]
            filt_data[:, ::-1] = scipy.signal.sosfilt(sos,
                                                      fir_buffer[:, ::-1],
                                                      axis=1,
                                                      zi=ic)[0][:, -n_samples:]

    filter_func = causal_filter if causal else acausal_filter

    return filter_func


if __name__ == '__main__':
    gc.disable()  # disable garbage collection

    thresh_ext = ThresholdExtraction()
    thresh_ext.run()

    gc.collect()  # run garbage collection
