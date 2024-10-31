#!/usr/bin/env python
# -*- coding: utf-8 -*-
# bin_multiple.py
import gc
import json
import logging
import time

import numpy as np
from brand import BRANDNode
from brand.redis import xread_sync


class BinThresholds(BRANDNode):

    def __init__(self):

        super().__init__()

        # set defaults
        self.parameters.setdefault('output_dtype', 'int8')

        # initialize parameters
        self.chan_per_stream = self.parameters['chan_per_stream']
        self.bin_size = self.parameters['bin_size']
        self.input_streams = self.parameters['input_streams']
        self.input_field = self.parameters['input_field']
        self.input_dtype = self.parameters['input_dtype']
        self.output_stream = self.parameters['output_stream']
        self.output_dtype = self.parameters['output_dtype']
        if 'xread_sync_field' in self.parameters:
            self.xread_sync_field = self.parameters['xread_sync_field'].encode()
        else:
            self.xread_sync_field = b'timestamps'
        if 'xread_sync_dtype' in self.parameters:    
            self.xread_sync_dtype = self.parameters['xread_sync_dtype']
        else:
            self.xread_sync_dtype = 'uint64'

        if not isinstance(self.chan_per_stream, list):
            self.chan_per_stream = [self.chan_per_stream] * len(self.input_streams)
        
        # Make dictionary of streams and channel slices
        self.channel_slices = {}
        ii = 0
        for stream, chans in zip(self.input_streams, self.chan_per_stream):
            self.channel_slices[stream.encode()] = slice(ii, ii + chans)
            ii += chans

        # initialize input stream entry data
        self.stream_dict = {name.encode(): '$' for name in self.input_streams}

        # handle data types
        # input data type
        if isinstance(self.input_dtype, str):
            self.input_dtype = [self.input_dtype] * len(self.input_streams)
        self.in_dtype = {
            name.encode(): np.dtype(dt)
            for name, dt in zip(self.input_streams, self.input_dtype)
        }
        # output data type
        self.out_dtype = np.dtype(self.output_dtype)

        # specify input field
        if isinstance(self.input_field, str):
            self.input_field = [self.input_field] * len(self.input_streams)
        self.in_field = {
            name.encode(): field.encode()
            for name, field in zip(self.input_streams, self.input_field)
        }

        logging.info(f'Reading from streams: {self.input_streams}')

        # define timing and sync keys
        self.time_key = 'ts'.encode()
        self.sync_key = 'sync'.encode()

        # initialize output stream entry data
        self.i = 0
        self.window = np.zeros(
            (np.sum(self.chan_per_stream), self.bin_size),
            dtype=self.out_dtype)

        self.output_entry = {}
        self.output_entry['samples'] = self.window.sum(axis=1).astype(
            self.out_dtype).tobytes()
        self.output_entry['i'] = self.i

        logging.info(f'Start spike binning from 1ms to {self.bin_size}ms...')

    def run(self):

        # count the number of entries we have read into the bin so far
        self.n_entries = 0

        while True:

            # reset number of entries into the bin
            self.n_entries = 0

            self.sync_entries = []

            # read `bin_size` samples from stream
            streams = xread_sync(self.r,
                                 self.stream_dict,
                                 block=0,
                                 sync_field=self.xread_sync_field,
                                 sync_dtype=self.xread_sync_dtype,
                                 count=self.bin_size)
            
            for stream in streams:
                stream_name, stream_entries = stream
                field = self.in_field[stream_name]
                in_type = self.in_dtype[stream_name]
                ch = self.channel_slices[stream_name]
                sync_entries_stream = []
                for i, (entry_id, entry_dict) in enumerate(stream_entries):
                    # load the input
                    self.window[ch, i] = np.frombuffer(entry_dict[field],
                                                       dtype=in_type).astype(
                                                           self.out_dtype)
                    # log sync for this entry
                    sync_entries_stream.append(
                        json.loads(entry_dict[b'sync'].decode()))
                self.sync_entries.append(sync_entries_stream)
                # update the xread ID
                self.stream_dict[stream_name] = entry_id

            # create sync dict from sync entries from input streams
            sync_dict = {}
            for stream in self.sync_entries:
                sync_entry_dict = stream[0]  # first entry from each stream
                for key in sync_entry_dict:
                    sync_dict[key] = sync_entry_dict[key]
            sync_dict_json = json.dumps(sync_dict)

            # write results to Redis
            self.output_entry[self.time_key] = np.uint64(
                time.monotonic_ns()).tobytes()
            self.output_entry[self.sync_key] = sync_dict_json
            self.output_entry['samples'] = self.window.sum(axis=1).astype(
                self.out_dtype).tobytes()
            self.output_entry['i'] = np.uint64(self.i).tobytes()

            self.r.xadd(self.output_stream, self.output_entry)

            self.i += 1


if __name__ == "__main__":
    gc.disable()

    # setup
    bin_thresholds = BinThresholds()

    # main
    bin_thresholds.run()

    gc.collect()
