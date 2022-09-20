#!/usr/bin/env python
# -*- coding: utf-8 -*-
# publisher.py
import gc
import logging
import sys
import time

import numpy as np
from brand import BRANDNode
from brand.timing import clock_nanosleep


class Publisher(BRANDNode):

    def __init__(self):

        super().__init__()

        # get parameters
        self.n_channels = self.parameters['n_channels']
        self.seq_len = self.parameters['seq_len']
        self.data_type = self.parameters['data_type']
        self.duration = self.parameters['duration']
        self.sample_rate = self.parameters['sample_rate']
        self.stop_graph_when_done = self.parameters['stop_graph_when_done']

        self.data = np.random.randn(self.n_channels,
                                    self.seq_len).astype(self.data_type)

        self.total_samples = int(self.duration * self.sample_rate)

        # print expected memory usage
        data_mem_size = self.data.size * self.data.itemsize  # bytes
        total_data_mem_size = data_mem_size * self.duration * self.sample_rate
        logging.info(f'Writing {data_mem_size / 2**10 :.4f} KB samples'
                     f' @ {self.sample_rate} Hz'
                     f' for {self.duration} seconds'
                     f' (total: {total_data_mem_size / 2**20 :.4f} MB)')

    def run(self):
        # initialize publisher dict
        publisher_entry = {'samples': self.data.tobytes()}
        # calculate the time between samples
        interval = 1_000_000_000 // self.sample_rate
        # send samples at a fixed sampling rate
        i = 0
        start_time = time.monotonic_ns()
        while i < self.total_samples:
            # write results to Redis
            publisher_entry['i'] = np.uint64(i).tobytes()
            publisher_entry['t'] = np.uint64(time.monotonic_ns()).tobytes()
            publisher_entry['timestamps'] = np.arange(
                i * 30, (i + 1) * 30, dtype=np.uint32).tobytes()
            self.r.xadd(self.NAME, publisher_entry)
            # update index and timestamp
            i += 1
            # sleep until the next loop
            clock_nanosleep(start_time + i * interval,
                            clock=time.CLOCK_MONOTONIC)

        if self.stop_graph_when_done:
            time.sleep(1)  # give the downstream nodes some time to process
            self.r.xadd('supervisor_ipstream', {'commands': 'stopGraph'})

    def terminate(self, *_):
        logging.info('SIGINT received, Exiting')
        sys.exit(0)


if __name__ == "__main__":
    gc.disable()

    node = Publisher()
    node.run()

    gc.collect()
