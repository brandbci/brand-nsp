


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




@njit('float32[:,:](float32[:,:], float32[:,:], float32[:,:], int64)')
def reref_neural_data_loop(coefs, neural_data, neural_data_reref, n_split):
    for i in range(0, neural_data.shape[0], n_split):
        end_idx = min(i + n_split, neural_data.shape[0])  # Prevent going past array bounds
        neural_data_reref[i:end_idx,:] = np.dot(coefs[i:end_idx,:], neural_data[:])

    return neural_data_reref

@njit('float32[:,:](float32[:,:], float32[:,:], float32[:,:], int64)')
def reref_neural_data(coefs, neural_data, neural_data_reref,n_split=0):
    neural_data_reref[:,:] = np.dot(coefs, neural_data[:])

    return neural_data_reref




@njit(['int16[:](float32[:,:],float32[:,:],float32[:,:],int16[:])',
       'int16[:](float32[:,:],float32[:,:],float64[:,:],int16[:])'])
def get_threshold_crossing(crossings,filt_buffer,thresholds,cross_now):                

    crossings[:, 1:] = ((filt_buffer[:, 1:] < thresholds) &
                                    (filt_buffer[:, :-1] >= thresholds))
    cross_now = (crossings.sum(axis=1)>=1).astype(np.int16)
    return cross_now








@njit('float32[:](float32[:,:], float32[:], boolean)')
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
    # read_times = np.empty((thresh_cal_len * samp_per_stream))

    i_start = 0
    for _, entry_data in entries:  # put it all into an array
        i_end = i_start + samp_per_stream
        read_arr[:, i_start:i_end] = np.reshape(
            np.frombuffer(entry_data[b'samples'], dtype=self.dtype),
            (self.n_channels_total, samp_per_stream))[self.n_range,:]
        # read_times[i_start:i_end] = np.frombuffer(
        #     entry_data[b'timestamps'], self.tdtype)
        i_start = i_end

    if self.causal:
        self.filter_func(read_arr, filt_arr, sos, zi)
    else:
        if self.demean:
            common_average_reference(read_arr, self.car_groups)
        filt_arr[:, :] = scipy.signal.sosfiltfilt(sos, read_arr, axis=1)

    thresholds = (thresh_mult *
                    np.sqrt(np.mean(np.square(filt_arr), axis=1))).reshape(
                        -1, 1)

    # log thresholds to database
    thresolds_enc = thresholds.astype(np.int16).tobytes()
    self.r.xadd(f'{self.NAME}_thresholds', {b'thresholds': thresolds_enc})

    logging.info('Calculated and set thresholds')
    return thresholds













def get_filter_func(demean, causal=False, use_fir=True):
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

    def causal_filter(data, filt_data, sos, zi, group_list):
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
        if demean:
            common_average_reference(data, group_list)
        filt_data[:, :], zi[:, :] = scipy.signal.sosfilt(sos,
                                                         data,
                                                         axis=1,
                                                         zi=zi)

    def acausal_filter(data,
                       filt_data,
                       rev_buffer,
                       sos,
                       zi,
                       group_list,
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
        if demean:
            common_average_reference(data, group_list)

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

def common_average_reference(data, group_list):
    """
    common average reference by group

    Parameters
    ----------
    data : array_like
        An 2-dimensional input array with shape
        [channel x time]
    group_list : list
        List of lists of channels grouped together across
        which to compute a common average reference
    """
    for g in group_list:
        data[g, :] -= data[g, :].mean(axis=0, keepdims=True)