


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




@jit(nopython=True)
def get_threshold_crossings(dat, thresholds):
    '''
    Get threshold crossings for all channels in a data window
    Args:
        dat         : 1ms neural data window [samples x channels] of type int16 or float32 
        thresholds  : precomputed threshold levels for all channels [channels] of type float64
    Returns:
        threshold_crossings : returns threshold crossings [channels] of type int16 for this dat window
    '''

    # get bool values for spikes in all channels, optimized for jit
    min_dat = np.zeros(shape=(dat.shape[-1]), dtype='float32')
    for ch in range(dat.shape[-1]):                # get min of samples in each channel
        min_dat[ch] = np.min(dat[:,ch])
    threshold_crossings = min_dat <= thresholds    # If min in a channel is less than a threshold, then it is a spike

    # convert bools to 0 and 1
    threshold_crossings = np.multiply(threshold_crossings,1)

    threshold_crossings = threshold_crossings.astype('int16')

    return threshold_crossings


# def get_adaptive_thresholds(mean_squared_new, mean):
#     self.mean_squared_new = np.mean(filt_buffer**2, axis=1)
#     self.mean_squared_last += (self.mean_squared_new - self.mean_squared_buffer[:,self.mean_squared_buffer_index])/self.rms_window_len
#     self.mean_squared_buffer[:,self.mean_squared_buffer_index] = self.mean_squared_new
#     self.mean_squared_buffer_index += 1
#     if self.mean_squared_buffer_index >= self.rms_window_len:
#         self.mean_squared_buffer_index = 0
#         self.mean_squared_buffer_full = True
#     self.root_mean_squared = np.sqrt(self.mean_squared_last)
#     if self.mean_squared_buffer_full:
#         self.thresholds = (self.thresh_mult * self.root_mean_squared).reshape(-1,1)
#         thresholds = self.thresholds



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
            
