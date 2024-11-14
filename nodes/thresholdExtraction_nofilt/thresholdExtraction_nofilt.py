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
from numba import jit

class ThresholdExtraction(BRANDNode):

    def __init__(self):
        super().__init__()

        # threshold multiplier, usually around -5
        self.thresh_mult = self.parameters['thresh_mult']
        # number of Redis packets to read on each iteration
        self.pack_per_call = self.parameters['pack_per_call']

        # parameters of the input stream
        self.input_stream = self.parameters['input_name']
        # number of samples per channel per redis entry
        self.samp_per_stream = self.parameters['input_samp_per_stream']
        # number of channels
        self.n_channels_total = self.parameters['input_chan_per_stream']

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

        # optional datatype
        if 'timestamp_data_type' in self.parameters:
            self.tdtype = self.parameters['timestamp_data_type']
        else:
            self.tdtype = np.uint32
        
        # use tracking id instead of NSP timestamp
        if 'use_tracking_id' in self.parameters:
            self.use_tracking_id = self.parameters['use_tracking_id']
        else:
            self.use_tracking_id = False


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

        # initialize adaptive threholding
        if 'adaptive_thresholds' in self.parameters:
            self.adaptive_thresholds = self.parameters['adaptive_thresholds']
        else:
            self.adaptive_thresholds = False
        if self.adaptive_thresholds:
            self.rms_window_len = self.parameters['adaptive_rms_window_len']
            self.adaptive_rms_stream = self.parameters['adaptive_rms_stream']
            self.mean_squared_buffer = np.zeros((self.n_channels, self.rms_window_len), dtype=np.float64)
            self.mean_squared_buffer_index = 0
            self.mean_squared_buffer_full = False
            self.mean_squared_last = np.zeros((self.n_channels), dtype=np.float64)
            self.mean_squared_new = np.zeros((self.n_channels), dtype=np.float64)
            self.root_mean_squared = np.zeros((self.n_channels), dtype=np.float64)
            logging.info(f"Adaptive spike thresholds enabled, using RMS computed over {self.samp_per_stream*self.rms_window_len} 30kHz samples")

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

    def load_thresholds_from_file(self, thresholds_file, tf_chans):
        try:
            with open(thresholds_file, 'r') as f:
                thresh_yaml = yaml.safe_load(f)
            if 'thresholds' in thresh_yaml:
                if tf_chans is None:
                    if len(thresh_yaml['thresholds']) == self.n_channels_total:
                        logging.info(
                            f'Loaded thresholds from {thresholds_file}')
                        return np.array(thresh_yaml['thresholds']).reshape(
                            -1, 1)
                    else:
                        raise ValueError(
                            f'Number of thresholds in {thresholds_file} '
                            f'({len(thresh_yaml["thresholds"])}) does not '
                            f'equal n_channels parameter {(self.n_channels_total)}')
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
                if len(thresholds) == self.n_channels_total:
                    logging.info(f'Loaded thresholds from the {stream} stream')
                    return thresholds.reshape(-1, 1)
                else:
                    raise ValueError(
                        f'Number of thresholds in the {stream} stream '
                        f'({len(thresholds)}) does not equal n_channels '
                        f'parameter {(self.n_channels_total)}')
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

    # @jit(nopython=True)
    def run(self):
        # get class variables

        input_stream = self.input_stream
        pack_per_call = self.pack_per_call
        samp_per_stream = self.samp_per_stream
        thresholds = self.thresholds

        # initialize arrays
        n_samp = samp_per_stream * pack_per_call
        # data_buffer = np.zeros((self.n_channels, n_samp), dtype=np.float32)
        filt_buffer = np.zeros((self.n_channels, n_samp), dtype=np.float32)
        crossings = np.zeros((self.n_channels, n_samp), dtype=np.float32)

        samp_times = np.zeros(n_samp, dtype=self.tdtype)

        # initialize stream entries
        cross_dict = {}
        filt_dict = {}
        rms_dict = {}
        coinc_dict = {}
        sync_dict = {self.sync_source_id: int(samp_times[0])}

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
                    filt_buffer[:, indStart:indEnd] = np.reshape(
                        np.frombuffer(entry_data[b'samples'],
                                      dtype=self.dtype),
                        (self.n_channels_total, samp_per_stream))[self.n_range,:]
                    if self.use_tracking_id:
                        samp_times[indStart:indEnd] = np.repeat(np.frombuffer(
                            entry_data[b'tracking_id'], self.tdtype), samp_per_stream)
                    else:
                        samp_times[indStart:indEnd] = np.frombuffer(
                            entry_data[b'timestamps'], self.tdtype)
                    indStart = indEnd

                # update key to be the entry number of last item in list
                input_stream_dict[input_stream] = entry_id


                samp_time_current = samp_times[:n_samp]

                # Adaptive spike thresholding
                if self.adaptive_thresholds:
                    # Compute MS for samples corresponding to current ms
                    self.mean_squared_new = np.mean(filt_buffer**2, axis=1)
                    # Update rolling MS iteratively using new and last stored sample
                    self.mean_squared_last += (self.mean_squared_new - self.mean_squared_buffer[:,self.mean_squared_buffer_index])/self.rms_window_len
                    # Store new MS in buffer, overrtiting oldest sample
                    self.mean_squared_buffer[:,self.mean_squared_buffer_index] = self.mean_squared_new
                    # Circular buffer
                    self.mean_squared_buffer_index += 1
                    if self.mean_squared_buffer_index >= self.rms_window_len:
                        self.mean_squared_buffer_index = 0
                        self.mean_squared_buffer_full = True
                    # Compute RMS
                    self.root_mean_squared = np.sqrt(self.mean_squared_last)
                    # If buffer has filled up, update thresholds using rolling RMS
                    if self.mean_squared_buffer_full:
                        self.thresholds = (self.thresh_mult * self.root_mean_squared).reshape(-1,1)
                        thresholds = self.thresholds

                sync_dict[self.sync_source_id] = int(samp_time_current[0])
                cross_dict[self.sync_key] = json.dumps(sync_dict)
                cross_dict[b'timestamps'] = samp_time_current[0].tobytes()
                # is there a threshold crossing in the last ms?
                crossings[:, 1:] = ((filt_buffer[:, 1:] < thresholds) &
                                    (filt_buffer[:, :-1] >= thresholds))
                cross_now = np.any(crossings, axis=1).astype(np.int16)

                # Redis
                p = self.r.pipeline()  # create a new pipeline

                # log timestamps
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
                
                if self.adaptive_thresholds:
                    # Update rms_dict for writing to Redis
                    rms_dict[self.sync_key] = json.dumps(sync_dict)  
                    rms_dict[b'timestamps'] = samp_time_current[0].tobytes()    
                    rms_dict[b'samples'] = self.root_mean_squared.astype(
                        np.float64).tobytes()  
                    rms_dict[b'thresholds'] = thresholds.astype(np.float64).tobytes() 
                    rms_dict[self.time_key] = time_now   
                    p.xadd(self.adaptive_rms_stream, rms_dict)             

                # write to Redis
                p.execute()

            elif len(xread_receive) == 0:
                logging.warning("No neural data has been received in the"
                                f" last {timeout} ms")

    # def terminate(self, sig, frame):
    #     if (hasattr(self, 'adaptive_thresholds') and 
    #             hasattr(self, 'thresholds') and 
    #             hasattr(self, 'thresholds_stream')):
    #         if (self.adaptive_thresholds and 
    #                 self.thresholds is not None and 
    #                 self.thresholds_stream is not None):
    #             self.r.xadd(self.thresholds_stream, {'thresholds': self.thresholds.tobytes()})
    #             logging.info(f'Logging latest adaptive thresholds to stream: \'{self.thresholds_stream}\'')
    #     return BRANDNode.terminate(self, sig, frame)

# Filtering functions


if __name__ == '__main__':
    gc.disable()  # disable garbage collection

    thresh_ext = ThresholdExtraction()
    thresh_ext.run()

    gc.collect()  # run garbage collection
