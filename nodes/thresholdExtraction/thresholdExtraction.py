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

        # parameters of the input stream
        self.input_params = self.parameters['input_stream']
        self.input_stream = self.input_params['name']
        # number of samples per channel per redis entry
        self.samp_per_stream = self.input_params['samp_per_stream']
        # number of channels
        self.n_channels = self.input_params['chan_per_stream']

        # build filtering pipeline
        self.filter_func, self.sos, self.zi = self.build_filter()

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
        # sampling rate of the input
        fs = self.parameters['input_stream']['samp_freq']

        # set up filter
        nyq = .5 * fs
        sos = scipy.signal.butter(but_order, [but_low / nyq, but_high / nyq],
                                  btype='bandpass',
                                  analog=False,
                                  output='sos')  # set up a filter
        # initialize the state of the filter
        zi_flat = scipy.signal.sosfilt_zi(sos)
        # so that we have the right number of dimensions
        zi = np.zeros((zi_flat.shape[0], self.n_channels, zi_flat.shape[1]))
        # filter initialization
        for ii in range(0, self.n_channels):
            zi[:, ii, :] = zi_flat

        # select the filtering function
        message = (f'Loading {but_order :d} order, '
                   f'[{but_low :d} {but_high :d}] hz bandpass causal filter')
        if demean:
            message += ' with CAR'
        logging.info(message)
        filter_func = get_filter_func(demean)

        return filter_func, sos, zi

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

        self.filter_func(read_arr, filt_arr, sos, zi)
        thresholds = (thresh_mult *
                      np.sqrt(np.mean(np.square(filt_arr), axis=1))).reshape(
                          -1, 1)
        logging.info('Thresholds are set')
        return thresholds

    def run(self):
        # get class variables
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
        crossings = np.zeros_like(data_buffer)
        samp_times = np.zeros(samp_per_stream * pack_per_call, dtype=np.uint32)

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
                self.filter_func(data_buffer, filt_buffer, sos, zi)

                # is there a threshold crossing in the last ms
                # find for each channel along the first dimension, keep dims,
                # pack into a byte object and put into the thresh crossings
                # dict
                sync_dict['nsp_idx'] = int(samp_times[0])
                cross_dict[b'sync'] = json.dumps(sync_dict)
                cross_dict[b'timestamps'] = samp_times[0].tobytes()
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
def get_filter_func(demean):
    """
    Get a function for filtering the data

    Parameters
    ----------
    demean : bool
        Whether to apply a common average reference before filtering
    """

    def filter_func(data, filt_data, sos, zi):
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

    return filter_func


if __name__ == '__main__':
    gc.disable()  # disable garbage collection

    thresh_ext = ThresholdExtraction()
    thresh_ext.run()

    gc.collect()  # run garbage collection
