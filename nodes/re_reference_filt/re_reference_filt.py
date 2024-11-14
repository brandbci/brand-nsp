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
import os

import numpy as np
import numba as nb
from numba import jit
import scipy
from brand import BRANDNode


class reReference(BRANDNode):

    def __init__(self):
        super().__init__()
        
        self.input_stream_name = self.parameters['input_stream_name']
        self.output_stream_name = self.parameters['output_stream_name']
        self.coefs_stream_name = self.parameters['coefs_stream_name']
        if 'input_dtype' in self.parameters:
            self.input_dtype = self.parameters['input_dtype'] 
        else:
            self.input_dtype = np.int16
        if 'output_dtype' in self.parameters:
            self.output_dtype = self.parameters['output_dtype'] 
        else:
            self.output_dtype = np.float64
        # self.samp_freq = self.parameters['samp_freq']
        self.chan_per_stream = self.parameters['chan_per_stream']
        self.samp_per_stream = self.parameters['samp_per_stream']
        self.chan_total = self.parameters['chan_total']
        self.start_channel = self.parameters['start_channel']   
        if 'neural_data_field' in self.parameters:
            self.neural_data_field = self.parameters['neural_data_field'] 
        else:
            self.neural_data_field = 'samples'
        if 'ts_field' in self.parameters:
            self.ts_field = self.parameters['ts_field'] 
        else:
            self.ts_field = 'BRANDS_time'
        
        self.nsp_ts_field = self.parameters.setdefault('nsp_ts_field', 'timestamps').encode()
        self.nsp_ts_dtype = self.parameters.setdefault('nsp_ts_dtype', 'uint64')

        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)

        self.neural_data = np.zeros((self.chan_per_stream, self.samp_per_stream), dtype=self.output_dtype)
        self.neural_data_reref = np.zeros((self.chan_per_stream, self.samp_per_stream), dtype=self.output_dtype)
        self.coefs = np.zeros((self.chan_per_stream, self.chan_per_stream), dtype=self.output_dtype)

        if 'n_split' in self.parameters:
            self.n_split = self.parameters['n_split'] 
        else:
            self.n_split = 16

        self.initialize_coefficients()
        self.init_filter_parameters()
        self.input_id = '$'
        self.output_dict = {}

        self.last_ts = np.array([0])

        logging.info(f"Parameters loaded. Reading from stream: {self.input_stream_name}. Writing to stream: {self.output_stream_name}")

    def init_filter_parameters(self):
        self.pack_per_call = self.parameters['pack_per_call']
        # whether to export the filtered data
        self.output_filtered = self.parameters['output_filtered']
        # enable acausal filtering
        self.acausal_filter = self.parameters['acausal_filter']
        self.causal = not self.acausal_filter
        # length of the buffer used for acausal filtering
        self.acausal_filter_lag = self.parameters['acausal_filter_lag']

        # build filtering pipeline
        if self.causal:
            self.filter_func, self.sos, self.zi = self.build_filter()
        else:
            (self.filter_func, self.sos, self.zi, self.rev_win,
             self.rev_zi) = self.build_filter()
            
        # which channels to use
        if 'neural_ch_range' in self.parameters:
            if len(self.parameters['neural_ch_range']) == 2:
                self.n_range = np.arange(
                    self.parameters['neural_ch_range'][0],
                    self.parameters['neural_ch_range'][1])
            else:
                logging.warning(
                    '\'neural_ch_range\' parameter should be length 2,'
                    ' attempting to use all neural channels')
                self.n_range = np.arange(0, self.n_channels_total)
        else:
            self.n_range = np.arange(0, self.n_channels_total)                               
        self.n_range = self.n_range.astype(int)
        self.n_channels = self.n_range.shape[0]

        if 'use_tracking_id' in self.parameters:
            self.use_tracking_id = self.parameters['use_tracking_id']
        else:
            self.use_tracking_id = False


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
            filter_func = get_filter_func(causal, use_fir=use_fir)

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


    def initialize_coefficients(self):

        entries = self.r.xrevrange(self.coefs_stream_name, '+', '-', count=1)

        if len(entries):
            logging.info(f"Coefficients entry found in Redis stream: {self.coefs_stream_name}")

            entry_id, entry_dict = entries[0] 

            self.coefs_all = np.frombuffer(entry_dict[b'channel_scaling'], dtype=np.float64).reshape((self.chan_total, self.chan_total))
            self.coefs = self.coefs_all[self.start_channel:self.start_channel+self.chan_per_stream, 
                                        self.start_channel:self.start_channel+self.chan_per_stream]

            if b'channel_unshuffling' in entry_dict:
                self.unshuffle_all = np.frombuffer(entry_dict[b'channel_unshuffling'], dtype=np.float64).reshape((self.chan_total, self.chan_total))
                logging.info(f"Unshuffling matrix loaded from stream.")
            else:
                self.unshuffle_all = np.eye(self.chan_total, dtype=np.float64)
                logging.info(f"No unshuffling matrix found. Assuming channels are in order.")
            
            self.unshuffle = self.unshuffle_all[self.start_channel:self.start_channel+self.chan_per_stream, 
                                                self.start_channel:self.start_channel+self.chan_per_stream]

            self.coefs = (np.eye(self.chan_per_stream) - self.coefs) @ self.unshuffle
            self.coefs = self.coefs.astype(self.output_dtype)

        else:
            logging.info(f"Coefficients entry not found in Redis stream: {self.coefs_stream_name}. Setting coefficients to compute mean of all channels.")

            self.coefs = np.eye(self.chan_per_stream) - np.ones((self.chan_per_stream, self.chan_per_stream))/self.chan_per_stream
            self.coefs.astype(self.parameters['output_dtype'])
        
    def run(self):


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

        pack_per_call = self.pack_per_call
        samp_per_stream = self.samp_per_stream
        n_samp = samp_per_stream * pack_per_call
        data_buffer = np.zeros((self.n_channels, n_samp), dtype=np.float32)
        filt_buffer = np.zeros_like(data_buffer)
        rev_buffer = np.zeros(
            (self.n_channels, self.acausal_filter_lag + n_samp),
            dtype=np.float32)
        samp_times = np.zeros(n_samp, dtype=self.tdtype)
        buffer_len = rev_buffer.shape[1]
        samp_times_buffer = np.zeros(buffer_len, dtype=self.tdtype)
        buffer_fill = 0  # how many samples have been read into the buffer



        while True:

            # read neural data from redis
            self.replies = self.r.xread({self.input_stream_name: self.input_id}, 
                                        block=0,
                                        count=1)

            self.input_id = self.replies[0][1][0][0]
            self.entry_data = self.replies[0][1][0][1]
            
            # read timestamps
            ts = np.concatenate([self.last_ts, np.frombuffer(self.entry_data[self.nsp_ts_field], dtype=self.nsp_ts_dtype).astype(int)])
            # check if timestamps are in order
            neg_time_diff = np.diff(ts) < 0
            if np.any(neg_time_diff):
                neg_ts = ts[1:][neg_time_diff]
                logging.warning(f"Timestamps {neg_ts} are not in order!!!")
            self.last_ts = ts[-1:]
            
            self.neural_data[:] = np.frombuffer(self.entry_data[self.neural_data_field.encode()], 
                                                dtype=self.input_dtype).reshape((self.chan_per_stream, self.samp_per_stream)).astype(self.output_dtype)

            self.n = 0
            while self.n < self.chan_per_stream:
                self.neural_data_reref[self.n:self.n+self.n_split,:] = np.dot(self.coefs[self.n:self.n+self.n_split,:], 
                                                                              self.neural_data[:])
                self.n += self.n_split


                self.neural_data_field == "samples"



            data_buffer = self.neural_data_reref
            indStart = 0
            for entry_id, entry_data in self.entry_data:
                indEnd = indStart + samp_per_stream

                if self.use_tracking_id:
                    samp_times[indStart:indEnd] = np.repeat(np.frombuffer(
                        entry_data[b'tracking_id'], self.tdtype), samp_per_stream)
                else:
                    samp_times[indStart:indEnd] = np.frombuffer(
                        entry_data[b'timestamps'], self.tdtype)
                indStart = indEnd


            # filter the data and find threshold times
            if self.causal:
                self.filter_func(data_buffer, filt_buffer, sos, zi)
            else:
                self.filter_func(data_buffer,
                                    filt_buffer,
                                    rev_buffer,
                                    sos=sos,
                                    zi=zi,
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






            # copy data to output dict
            for key in self.entry_data.keys():
                if key.decode() != self.neural_data_field and key.decode() != self.ts_field:
                    self.output_dict[key] = self.entry_data[key]
                elif key.decode() == self.ts_field:
                    self.output_dict[self.ts_field] = np.uint64(time.monotonic_ns()).tobytes()
                elif key.decode() == self.neural_data_field:
                    self.output_dict[self.neural_data_field] = filt_buffer.astype(self.output_dtype).tobytes()
            
            # write to Redis
            self.r.xadd(self.output_stream_name, self.output_dict)


# Filtering functions
def get_filter_func(causal=False, use_fir=True):
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
        group_list : list
            List of lists of channels grouped together across
            which to compute a common average reference
        """

        filt_data[:, :], zi[:, :] = scipy.signal.sosfilt(sos,
                                                         data,
                                                         axis=1,
                                                         zi=zi)

    def acausal_filter(data,
                       filt_data,
                       rev_buffer,
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




if __name__ == '__main__':
    gc.disable()  # disable garbage collection

    reref = reReference()
    reref.run()

    gc.collect()  # run garbage collection
