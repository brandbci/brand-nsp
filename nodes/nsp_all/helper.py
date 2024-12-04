


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
from numba import jit, njit, float64, boolean, int16, int32, prange
import time




@njit('float64[:,:](float64[:,:], float64[:,:], float64[:,:], int32, int32)')
def reref_neural_data(coefs, neural_data, neural_data_reref, n_split):
    n = 0
    while n < neural_data.shape[0]:
        neural_data_reref[n:n+n_split,:] = np.dot(coefs[n:n+n_split,:], neural_data[:])
        n += n_split

    return neural_data_reref


# @njit('float64[:,:](float64[:,:], float64[:,:])')
# def numba_dot(A, B):
#     return np.dot(A, B)


@njit('void(float64[:,:], float64[:],int16[:])')
def get_threshold_crossing(filt_buffer, thresholds, cross_now):
    num_channels, buffer_size = filt_buffer.shape
    # cross_now = np.zeros(num_channels, dtype=np.int16)
    
    for i in range(num_channels):
        threshold = thresholds[i]
        for j in range(1, buffer_size):
            if filt_buffer[i, j] < threshold and filt_buffer[i, j-1] >= threshold:
                cross_now[i] = 1
                break
    
    return cross_now



@njit('void(float64[:,:], float64[:,:], boolean)')
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
    
    return power_buffer



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
