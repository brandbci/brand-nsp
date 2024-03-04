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

from brand import BRANDNode


# @nb.jit
# def matmul(A, B):
    
#     m,n = A.shape
#     p = B.shape[1]

#     C = np.zeros((m,p))

#     for i in range(0,m):
#         for j in range(0,p):
#             for k in range(0,n):
#                 C[i,j] += A[i,k]*B[k,j] 
#     return C

#     # for ch in range(self.chan_per_stream):
#     #     self.neural_data_reref[ch,:] = np.matmul(self.coefs[ch,:], self.neural_data)
#     # return np.matmul(A, B)


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

        self.input_id = '$'
        self.output_dict = {}

        logging.info(f"Parameters loaded. Reading from stream: {self.input_stream_name}. Writing to stream: {self.output_stream_name}")

        # os.environ["OMP_NUM_THREADS"] = "2"
        # logging.info(f"Num threads: {n_threads}")

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

    # def calc(self):
    #     return self.matmul

    # @staticmethod
    # @jit(nopython=True)
    # def matmul(A, B):
        
    #     m,n = A.shape
    #     p = B.shape[1]

    #     C = np.zeros((m,p))

    #     for i in range(0,m):
    #         for j in range(0,p):
    #             for k in range(0,n):
    #                 C[i,j] += A[i,k]*B[k,j] 
    #     return C

    #     # for ch in range(self.chan_per_stream):
    #     #     self.neural_data_reref[ch,:] = np.matmul(self.coefs[ch,:], self.neural_data)
    #     # return np.matmul(A, B)

        
    def run(self):

        while True:

            # read neural data from redis
            self.replies = self.r.xread({self.input_stream_name: self.input_id}, 
                                        block=0,
                                        count=1)

            self.input_id = self.replies[0][1][0][0]
            self.entry_data = self.replies[0][1][0][1]

            # self.input_id, entry_data = entries[0]
            
            self.neural_data[:] = np.frombuffer(self.entry_data[self.neural_data_field.encode()], 
                                                dtype=self.input_dtype).reshape((self.chan_per_stream, self.samp_per_stream)).astype(self.output_dtype)

            # self.t_start = time.monotonic_ns()

            # self.neural_data_reref[:] = self.coefs @ self.neural_data.astype(self.output_dtype)
            # self.neural_data_reref[:] = np.matmul(self.coefs, self.neural_data)
            # self.neural_data_reref[:] = np.dot(self.coefs, self.neural_data)

            self.n = 0
            while self.n < self.chan_per_stream:
                self.neural_data_reref[self.n:self.n+self.n_split,:] = np.dot(self.coefs[self.n:self.n+self.n_split,:], 
                                                                              self.neural_data[:])
                self.n += self.n_split

            # for ch in range(self.chan_per_stream):
            #     self.neural_data_reref[ch,:] = np.matmul(self.coefs[ch,:], self.neural_data)

            # self.t_end = time.monotonic_ns()

            # if self.t_end - self.t_start > 900000:
            #     logging.info(f"Matrix operation duration (ns): {self.t_end - self.t_start}")

            # copy data to output dict
            for key in self.entry_data.keys():
                if key.decode() != self.neural_data_field and key.decode() != self.ts_field:
                    self.output_dict[key] = self.entry_data[key]
                elif key.decode() == self.ts_field:
                    self.output_dict[self.ts_field] = np.uint64(time.monotonic_ns()).tobytes()
                elif key.decode() == self.neural_data_field:
                    self.output_dict[self.neural_data_field] = self.neural_data_reref.astype(self.output_dtype).tobytes()
            
            # write to Redis
            self.r.xadd(self.output_stream_name, self.output_dict)

if __name__ == '__main__':
    gc.disable()  # disable garbage collection

    reref = reReference()
    reref.run()

    gc.collect()  # run garbage collection
