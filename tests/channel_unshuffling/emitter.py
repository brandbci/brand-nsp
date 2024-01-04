
import gc
import json
import logging
import signal
import time
from copy import deepcopy
import os
import redis
import yaml

import numpy as np
import numba as nb
from numba import jit



class Emitter():

    def __init__(self):
        super().__init__()
        
        self.output_stream_name = 'nsp_neural'

        self.chan_per_stream = 256
        self.samp_per_stream = 30

        self.sampling_time = 0.001

        # terminate on SIGINT
        # signal.signal(signal.SIGINT, self.terminate)

        
    def run(self):

        # connect to redis
        self.r = redis.Redis()

        # add rereference stream entry
        participant = 'sim256'
        session = '2024-01-03'
        block = f'{participant}_240103_030'
        path = f'/samba/data/{participant}/{session}/RawData/thresh_norm/{block}.yaml'
        with open(path, 'r') as file:
            yaml_dict = yaml.safe_load(file)
        reref_param = np.array(yaml_dict['rereference_parameters'])
        unshuffle_matrix = np.array(yaml_dict['channel_unshuffling'])

        self.reref_dict = {}
        self.reref_dict['channel_scaling'] = reref_param.astype(np.float64).tobytes()
        # self.reref_dict['channel_scaling'] = np.zeros((256,256)).astype(np.float64).tobytes()
        self.reref_dict['channel_unshuffling'] = unshuffle_matrix.astype(np.float64).tobytes()
        self.r.xadd(b'rereference_parameters', self.reref_dict)

        print(unshuffle_matrix.sum())

        while True:

            # self.data = np.arange(0, self.chan_per_stream).reshape((self.chan_per_stream,1)) + 1
            self.data = np.array([63, 64, 62, 61, 59, 58, 60, 54, 57, 50, 53, 49, 52, 45, 55, 44, 56, 
                                  39, 51, 43, 46, 38, 48, 37, 47, 36, 42, 35, 41, 34, 40, 33, 96, 90, 
                                  95, 89, 94, 88, 93, 87, 92, 82, 86, 81, 91, 77, 85, 83, 84, 78, 80, 
                                  73, 79, 74, 75, 76, 71, 72, 68, 69, 66, 70, 65, 67, 128, 120, 127, 119, 
                                  126, 118, 125, 117, 124, 116, 123, 115, 122, 114, 121, 113, 112, 111, 109, 
                                  110, 107, 108, 106, 105, 104, 103, 102, 101, 100, 99, 97, 98, 32, 30, 31, 29, 
                                  28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 16, 17, 7, 15, 6, 14, 5, 13, 4, 
                                  12, 3, 11, 2, 10, 1, 9, 8, 191, 192, 190, 189, 187, 186, 188, 182, 185, 178, 
                                  181, 177, 180, 173, 183, 172, 184, 167, 179, 171, 174, 166, 176, 165, 175, 164, 
                                  170, 163, 169, 162, 168, 161, 224, 218, 223, 217, 222, 216, 221, 215, 220, 210, 
                                  214, 209, 219, 205, 213, 211, 212, 206, 208, 201, 207, 202, 203, 204, 199, 200, 
                                  196, 197, 194, 198, 193, 195, 256, 248, 255, 247, 254, 246, 253, 245, 252, 244, 
                                  251, 243, 250, 242, 249, 241, 240, 239, 237, 238, 235, 236, 234, 233, 232, 231, 
                                  230, 229, 228, 227, 225, 226, 160, 158, 159, 157, 156, 155, 154, 153, 152, 151, 
                                  150, 149, 148, 147, 146, 144, 145, 135, 143, 134, 142, 133, 141, 132, 140, 131,
                                  139, 130, 138, 129, 137, 136]).reshape((self.chan_per_stream,1))     
            
            self.data = np.tile(self.data, (1, self.samp_per_stream))

            self.output_dict = {b'samples': self.data.astype(np.int16).tobytes()}
            self.r.xadd(self.output_stream_name, self.output_dict)

            time.sleep(self.sampling_time)

if __name__ == '__main__':
    gc.disable()  # disable garbage collection

    emit = Emitter()
    emit.run()

    gc.collect()  # run garbage collection
