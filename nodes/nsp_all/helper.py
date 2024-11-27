


import numpy as np
import scipy
import random
import scipy.ndimage
import scipy.signal
import scipy.io
import json
import json
import yaml
import pickle
import sys
import numpy as np
from numba import jit, njit, float64, boolean, int16, int64, prange
import time





@njit(int16[:](float64[:,:], float64[:]))
def get_threshold_crossing(filt_buffer, thresholds ):
    num_channels, buffer_size = filt_buffer.shape
    cross_now = np.zeros(num_channels, dtype=np.int16)
    
    for i in range(num_channels):
        threshold = thresholds[i]
        for j in range(1, buffer_size):
            if filt_buffer[i, j] < threshold and filt_buffer[i, j-1] >= threshold:
                cross_now[i] = 1
                break
    
    return cross_now



@njit(float64[:](float64[:,:], float64[:,:], boolean))
def get_spike_bandpower(filt_buffer, power_buffer, logscale=False):
    num_channels, buffer_size = filt_buffer.shape
    
    for i in range(num_channels):
        sum_squared = 0.0
        for j in range(buffer_size):
            if logscale:
                sum_squared += 10* np.log10(filt_buffer[i,j] * filt_buffer[i,j])
            else:
                sum_squared += filt_buffer[i,j] * filt_buffer[i,j]
        
        mean_power = sum_squared / buffer_size
        power_buffer[i] = mean_power
            
