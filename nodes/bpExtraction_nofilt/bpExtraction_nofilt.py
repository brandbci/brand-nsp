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

        # whether to log-transform the band power
        self.logscale = self.parameters.get('logscale', False)

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



        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)

    
    def run(self):
        # get class variables

        input_stream = self.input_stream
        pack_per_call = self.pack_per_call
        samp_per_stream = self.samp_per_stream

        # initialize arrays
        n_samp = samp_per_stream * pack_per_call
       
        filt_buffer = np.zeros((self.n_channels, n_samp), dtype=np.float32)
        power_buffer = np.zeros(self.n_channels, dtype=np.float32)
        samp_times = np.zeros(n_samp, dtype=self.tdtype)

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


                # convert to power
                if self.logscale:
                    power_buffer[:] = 10 * np.log10(
                        np.square(filt_buffer)).mean(axis=1)
                else:
                    power_buffer[:] = np.square(filt_buffer).mean(axis=1)

                # set sync index
                samp_time_current = samp_times[:n_samp]

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



if __name__ == '__main__':
    gc.disable()  # disable garbage collection

    bp_ext = bpExtraction()
    bp_ext.run()

    gc.collect()  # run garbage collection
