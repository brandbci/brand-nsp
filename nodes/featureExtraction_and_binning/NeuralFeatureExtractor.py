# ------------------------
# Functions for Neural feature extraction, pre-processing and read/write utilities
# Maitreyee Wairagkar, July 2023 [mwairagkar@ucdavis.edu]
# Created    : 2023-07-14
# Last update: 2024-01-30
# ------------------------

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
from numba import jit
import sys

# --------------------------------------------------------------------------------
# Functions to preprocess neural data and extract features 
# --------------------------------------------------------------------------------

def build_filter(filt_order, lo_cut, hi_cut, fs):
    '''
    Build filter coefficients for a non-causal Butterworth bandpass filter
    Args:
        filt_order: order of the butterworth filter (4th order filter preferred)
        lo_cut    : lower cutoff frequency
        hi_cut    : upper cutoff frequency
        fs        : sampling frequency
    Returns:
        b,a       : lists of filter coefficients
    
    '''
    #Filter parameters are fixed to match online featureExtraction

    # Get filter parameters for non-causal Butterworth filter
    [b,a] = scipy.signal.butter(filt_order, [lo_cut, hi_cut], btype='bandpass', 
                                analog=False, output='ba', fs=fs)
        
    return b,a


@jit(nopython=True)
def pad_data_for_filtering(dat, n_samples, dat_len):
    ''' 
    This function pads data for accurate filtering of as little as 1ms neural window
    Args:
        dat      : [samples x channels] data buffer of type int16 (could include historic data), should atleast be 1ms
        n_samples: number of samples in 1ms window
        dat_len  : length of the data window to be filtered (without historic data) 
    Returns:
        dat_filt_buff : A buffer of padded signal for filtering [samples x channels] of type int16
        pad_len     : Length of padding at the beginning of the data buffer (can include historic data or mean padded data atthe beginning of neural data window)
        
    '''
    
    # First concatenate: (1) dat (which may contain a few ms of hostoric data), 
    # (2) a small portion of flipped data to be filtered to avoid discontinuities at the edge, whilst preserving freq info
    # (3) mean padding of 1ms at the end 
    # (4) If no historic data is provided, add mean padding of 1ms at the beginning
    
    hist_dat_len = dat.shape[0] - dat_len   # Determine how much historic data was added to the input data buffer
    flip_len     = int(n_samples/5)         # Length of data to flip and append to the data buffer (0.2 or 0.25 ms is sufficient)
    l            = 0                        # A counter to keep track of concatenation indices 

    # Determine the length of mean padding to be added at the beginning.
    if hist_dat_len == 0:     # If no historic data is provided, mean padd start len is 1ms
        mean_pad_start_len = n_samples
    else:                     # If historic data is provided, mean pad start len is 0, i.e. no padding
        mean_pad_start_len = 0
        dat          = dat[flip_len:]           # Remove the flip_len amount of data from historic data at the beginning (this helps in speeding the filtering for 1ms windows)
        hist_dat_len = dat.shape[0] - dat_len   # Recalculate the historic data length after removing the flip_len amount of data from the beginning
    
    # Flip each channel along X axis and append to current data to increase dat len without discontinuities
    flipped_dat = dat[::-1] 
    
    # Mean padding of 1ms to add at the end of the data (and at the beginning if required)
    mean_val = np.zeros(shape=(dat.shape[-1]), dtype='int16')
    for ch in range(dat.shape[-1]):
        mean_val[ch] = np.round(np.mean(dat[:,ch]))
    mean_pad = np.ones(shape=(n_samples, dat.shape[-1]), dtype='int16') * mean_val
    
    # Create a buffer to hold concatenated data for filtering * Note that all elements in the concatenated data should be of type int16
    dat_filt_buff = np.zeros(shape=(mean_pad_start_len 
                                    + dat.shape[0] 
                                    + flip_len
                                    + mean_pad.shape[0], dat.shape[-1]), dtype='int16')
    
    # Concatenate all required data and paddings (this is faster than np.pad or np.concatenate functions)
    dat_filt_buff[:mean_pad_start_len, :] = mean_pad[:mean_pad_start_len, :]
    l = l + mean_pad_start_len
    dat_filt_buff[l:l+dat.shape[0],:] = dat
    l = l + dat.shape[0] 
    dat_filt_buff[l:l+flip_len,:] = flipped_dat[:flip_len]
    l = l + flip_len
    dat_filt_buff[l:,:] = mean_pad
    
    # Compute actual padding length at the beginning (including historic data and mean padding)
    pad_len = hist_dat_len + mean_pad_start_len 
    
    return dat_filt_buff, pad_len


def filter_signal(dat, n_samples, b, a, dat_len):
    '''
    Filter as little as 1ms window with a few ms of history (used in online filtering)
    Args:
        dat      : [samples x channels] data buffer of type int16 (could include historic data), should atleast be 1ms
        n_samples: number of samples in 1ms window
        b, a     : filter coefficients
        dat_len  : length of the data window to be filtered (without historic data) 
    Returns:
        filt_sig : returns filtered window [samples x channels] of type int16 and length dat_len (excludes historic data or any padding added)
    '''

    # Get paded dat buffer for filtering
    dat_filt_buff, pad_len = pad_data_for_filtering(dat, n_samples, dat_len)
    
    # Filter the 2D signal - make sure to use correct axis
    dat = scipy.signal.filtfilt(b, a, dat_filt_buff, axis=0)
   
    # remove padded portions from the filtered data
    filt_sig = dat[pad_len:pad_len+dat_len,:].astype('int16')

    return filt_sig


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


@jit(nopython=True)
def get_spike_bandpower(dat, clip_thresh):
    '''
    Get spike band power in the given (1ms) data window by squaring and averaging the amplitude of each channel
    Args:
        dat         : 1ms neural data window [samples x channels] of type int16 or float32
        clip_thresh : threshold to clip the spike power to avoid outliers
    Returns:
        spike_pow   : returns spike power [channels] of type float32 for this dat window
    '''

    # up the dtype to avoid overflow during squaring
    dat = dat.astype(np.float32) 

    # get spike power in each channel by squaring samples in it and averaging. Optimized for jit
    spike_pow = np.zeros(shape=dat.shape[-1], dtype='float32')
    for ch in range(dat.shape[-1]):
        spike_pow[ch] = np.mean(np.square(dat[:,ch]))

    # clip max spike power to a pre-defined clip threshold
    np.clip(spike_pow, None, clip_thresh, out=spike_pow)

    return spike_pow


@jit(nopython=True)
def car(dat, n_arrays, n_electrodes_per_array):
    '''
    Common averaging referencing seperately for each array
    Args:
        dat         : neural data window [samples x channels] of type int16
        n_arrays    : number of arrays
        n_electrodes_per_array: number of electrode per array. Assuming equal number of electrodes per array
    Returns:
        dat         : returns CAR referenced data window [samples x channels] of type float32
    '''
    
    dat = dat.astype('float32')
    n_electrodes = n_electrodes_per_array

    for array in range(n_arrays):
        
        this_array = np.arange(n_electrodes*array,n_electrodes*(array+1))

        dat_this_array = np.ascontiguousarray(dat[:,this_array])

        # Get mean of all channels per sample
        this_mean = np.zeros(shape=(dat.shape[0],1), dtype='float32')
        for sample in range(dat.shape[0]):
            this_mean[sample,0] = np.mean(dat_this_array[sample,:])
    
        dat[:,this_array] =  dat_this_array - this_mean
        
    return dat


@jit(nopython=True)
def lrr(dat, n_arrays, n_electrodes_per_array, lrr_weights):
    '''
    This function performs linear regression referencing (LRR) for each array.
    LRR is a referencing technique that eliminates noise or other common signals among neural channels.
    Unlike common average referencing (CAR), LRR does not assume that all channels on an array are equally noisy.
    For more information, see (Young et al, 2018): https://iopscience.iop.org/article/10.1088/1741-2552/aa9ee8/pdf
    Args:
        dat         : neural data window [samples x channels] of type int16
        n_arrays    : number of arrays
        n_electrodes_per_array: number of electrode per array. Assuming equal number of electrodes per array
        lrr_weights : A 3D matrix with 2D weight coefficients for each of the arrays for LRR rereferencing of type float32
    Returns:
        dat         : returns LRR referenced data window [samples x channels] of type float32
    '''
    
    dat = dat.astype('float32')
    lrr_weights = lrr_weights.astype('float32')
    n_electrodes = n_electrodes_per_array

    for array in range(n_arrays):

        this_array = np.arange(n_electrodes*array,n_electrodes*(array+1))

        # get weights for this array
        this_array_coeffs = np.ascontiguousarray(lrr_weights[array])

        # perform LRR by subtracting weighted array signal from each channel
        dat_this_array = np.ascontiguousarray(dat[:,this_array])
        dat[:,this_array] = dat_this_array - np.dot(dat_this_array, this_array_coeffs)

    return dat


@jit(nopython=True)
def compute_thresholds(dat, thresh_mult):
    '''
    Calculate RMS thresholds for spike detection from broadband filtered neural data (fast implementation using jit)
    Args:
        dat         : neural data window [samples x channels] of type int16 or float32
        thresh_mult : threshold multiplier (usually -4.5)
    Returns:
        threshold   : returns threshold [channels] of type float64 
    '''

    threshold = np.zeros(dat.shape[1], dtype='float64')

    for ch in range(dat.shape[1]):
        rms = np.sqrt(np.mean(np.square(dat[:,ch]))) # RMS preferred over std as this does not assume zero mean
        threshold[ch] = thresh_mult*rms              # threshold multiplier is -4.5
    
    return threshold


@jit(nopython=True) 
def lstsq_pseudoinverse(dat, remaining_idx, ch_idx):
    '''
    Solve the fast implementation of least squares with Moore-Penrose inverse
    to estimate W = inv(X'X)X'Y
    where W is regression coefficient vector to estimate Y from X
    This is a supplimentary function to compute LLR weights of the neural data
    Using jit nopython decorator requires data in np.float32 
    dat           :input neural data matrix [samples x channels]
    ch_idx        :index of the current channel (gives prediction vector Y of the equation)
    remaining_idx :indices of the remaining channels that are used for predicting the weights (gives X matrix)
    return        :W output of regression
    '''
    X = np.ascontiguousarray(dat[:,remaining_idx])
    Y = np.ascontiguousarray(dat[:,ch_idx])
    
    W = np.linalg.solve(X.T.dot(X), X.T.dot(Y)) 
    
    return W


@jit(nopython=True)
def getLRRWeights_optimized(dat, fs, max_seconds=45):
    '''
    Calculate LRR weights for one array
    Optimised for numba jit decorator for speeding the implementation
    Args:
        dat         : neural data window [samples x channels] of type int16
        fs          : sampling frequency
        max_seconds : maximum number of seconds to use for weight calculation
    Returns:
        ref_mat     : returns LRR weight matrix [channels x channels] of type float32
    '''

    these_chans = np.arange(dat.shape[1]) # channels to use for LRR. Assumes that all channels for one array are provided.
    ref_mat     = np.zeros((these_chans.shape[0], these_chans.shape[0])).astype(np.float32) # pre-allocate weight matrix
    
    '''
    Subsample data to use for LRR weight calculation.
    Randomize the order of the data on the time axis to avoid biasing the LRR weights.
    Only use up to max_seconds of data. If the data is less than max_seconds, use all of it.
    Note that subsampling and randomization is only done for weight calculation. LRR is later applied to all data.
    '''
    max_idx       = max_seconds*fs
    up_to_idx     = np.min(np.array([max_idx, dat.shape[0]]))
    rand_idx      = np.random.permutation(np.arange(dat.shape[0]))
    use_idx       = rand_idx[0:up_to_idx]
    subsample_dat = dat[use_idx,:]
    
    for chan_idx in range(len(these_chans)): #these_chans: # loop through channels

        '''
        Here is where the LRR weights actually get calculated.
        For each channel, we are calculating the weights of all other channels to be later subtracted.
        We do this by solving the equation:
        Y = X*W
        where Y is the data from the channel we are calculating weights for, X is the data from all other channels, and W is the weight matrix.
        We solve for W using least squares. *** 
        W = inv(X.T X) X.T Y
        Repeat this for every channel. Resultant weight matrix is ref_mat, of size (n_channels, n_channels).
        '''
        
        # Get a list of all chanel indices excluding the current one
        pred_idx = these_chans[np.arange(these_chans.shape[0])!=chan_idx]
                
        # Solve the optimised least squares to get weights for this channel
        weights  = lstsq_pseudoinverse(subsample_dat.astype(np.float32),  pred_idx, chan_idx)
        
        # Add the weights to the larger weight matrix of all channels in appropriate positions, 
        # leaving space for the current channel where the weight is zero - this has faster implementation
        ref_mat[chan_idx, :chan_idx]   = weights[:chan_idx] 
        ref_mat[chan_idx, chan_idx+1:] = weights[chan_idx:] 
        
    return ref_mat


@jit(nopython=True)
def get_lrr_weights_wrapper(dat, n_arrays, n_electrodes_per_array, fs, max_seconds=45, toPrint=False, module_name='LRR'):
    '''
    This function calculates LRR weights for each array.
    LRR (linear regression referencing) is a referencing technique that eliminates noise or other common signals among neural channels.
    Unlike common average referencing (CAR), LRR does not assume that all channels on an array are equally noisy.
    For more information, see (Young et al, 2018): https://iopscience.iop.org/article/10.1088/1741-2552/aa9ee8/pdf
    Args:
        dat                   : neural data block (n_samples x n_channels) of type int16 (could be downsampled)
        n_arrays              : number of arrays
        n_electrodes_per_array: number of electrodes per array
        fs                    : sampling frequency (account for downsampling)
        max_seconds           : maximum number of seconds to use for LRR calculation
        toPrint               : boolean flag to print progress
        module_name           : string of name of the module to be printed
    Returns:
        lrr_weights           : LRR weights for each array (n_arrays x n_electrodes_per_array x n_electrodes_per_array) of type float32
    '''
    n_electrodes = n_electrodes_per_array                   # electrodes per array
    lrr_weights = np.zeros(shape=(n_arrays, n_electrodes, n_electrodes), dtype='float32') # weight matrix to store LRR weights for each array

    # loop through arrays
    for array in range(n_arrays):

        this_array = np.arange(n_electrodes*array,n_electrodes*(array+1))
        
        # calculate LRR weights and append them to list
        this_array_coeffs  = getLRRWeights_optimized(dat[:,this_array], fs, max_seconds)
        lrr_weights[array] = this_array_coeffs

        if toPrint:
            print(f'[{module_name}]: LRR weights computed for array #{array+1}')

    return lrr_weights


def bin_data(dat, bin_type):
    '''
    Bin the given window of data by averaging or summing
    Args:
        dat         : neural data window [samples x channels] of type int16 or float32
        bin_type    : 0 for averaging, 1 for summing
    Returns:
        binned_dat  : returns binned data [channels] of type float32 or int16
    '''

    if bin_type == 0: # bin_type = 0 for averaging the values in the bin
        binned_dat = np.mean(dat, axis=0)
        binned_dat = binned_dat.astype('float32')

    elif bin_type == 1: # sum the values in the bin, used for getting threshold count
        binned_dat = np.sum(dat, axis=0)
        binned_dat = binned_dat.astype('int16')

    return binned_dat


def halfgaussian_kernel1d(sigma, truncate=4.0):
    '''
    Compute weights for Half-Gaussian kernal window.
    '''
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    radius = int(truncate * sd + 0.5)
    
    sigma2 = sigma * sigma
    x      = np.arange(-radius, 0+1)
    phi_x  = np.exp(-0.5 / sigma2 * x ** 2)
    
    weights = phi_x / phi_x.sum()

    return weights


def exponential_kernel1d(win_len=15, sigma=3.0):
    '''
    Compute weights for exponential (polynomial) kernal window.
    '''
    radius = int(win_len)
    
    x     = np.arange(radius)
    phi_x = x ** sigma 
    
    weights = phi_x/np.sum(phi_x)
    
    return weights


def sigmoid_kernel1d(win_len=15, sigma=0.5):
    '''
    Compute weights for sigmoid kernal window.
    '''

    radius = win_len//2
    
    x = np.arange(-radius, radius+1)
    
    phi_x = 1/(1+np.exp(-sigma*x))
    weights = phi_x/np.sum(phi_x)
    
    return weights


def get_kernel(kernel_type, sigma, kernel_len):
    '''
    This function returns a causal kernel of a given type and parameters
    Args:
        kernel_type: smoothing kernal type
        sigma      : Kernel shape parameter
        kernel_len : Kernel length in bins. This will determine how many past bins will be used for smoothing
    Returns:
        kernel_window : kernel window
    '''

    kernel_window = np.ones([kernel_len,])/kernel_len

    if kernel_type == 'half_gaussian':
        kernel_window = halfgaussian_kernel1d(sigma)

    elif kernel_type == 'exponential':
        kernel_window = exponential_kernel1d(kernel_len, sigma)

    elif kernel_type == 'sigmoid':
        kernel_window = sigmoid_kernel1d(kernel_len, sigma)

    kernel_window = kernel_window[:,np.newaxis]

    return kernel_window


def preprocess_bin(dat):
    '''
    Log transform the data to get normal distribution
    dat: neural binned data [samples x channels]
    '''
    dat = np.log(dat + 0.0001, dtype='float32') # add a small number to avoid log of zero
    
    return dat


def zscore(dat, mean, std):
    '''
    Z-score normalization of binned data
    Args:
        dat : neural binned data [samples x channels]
        mean: mean of the data
        std : standard deviation of the data
    Returns:
    '''

    mean = np.array(mean)
    std  = np.array(std)

    norm_dat = (dat- mean)/ (std+0.0001) 

    return norm_dat


def augment_neural_data(dat, param):
    '''
    Augment the data by adding three types of noise - channel drop, random noise, offset and cumulative noise
    If data has both spikepow and threshold crossings, then these must be concatenated in dat
    Note that mean and std are taken from Willett et al. 
    Note that they smooth the data after adding noise, I add noise to the smoothed neural data and then smooth it a bit again
    Args:
        dat: single trial [smooth bins x channels] here channels means features, i.e. spikepow chs + thresh cross chs
        param: global parameters dictionary
    Returns:
        dat: single trial augmented with noise
    '''
    
    num_channels = dat.shape[-1] # half of these channels are spike pow and the other half are thresh cros
    
    # chanel dropout - TODO: check this because 0 value has a meaning so channel drop is not currently used
    channels_to_drop = random.sample(range(num_channels),int(num_channels/6))
    #dat[:,channels_to_drop] = 0.0001
    
    # White noise on all channels
    whiteMu = 0.0
    whiteStd = 1.2 
    gaussian_noise = np.random.normal(whiteMu, whiteStd, size=dat.shape).astype('float32')

    # Add constant offset noise in each channel independently
    constantOffsetStd = 0.6
    constantOffsetMu = 0
    aug_factor_for_thresh = 0.67 # Constant offsect is changed by this factor for thresh cross so these features are not exactly the same as spike pow
    constant_offset = np.random.normal(constantOffsetMu, constantOffsetStd, size=(1,int(num_channels/2)))# only get offsets for spike pow features
    constant_offset_for_spikes = np.matlib.repmat(constant_offset, dat.shape[0], 1)
    constant_offset_for_thresh = np.matlib.repmat(constant_offset*aug_factor_for_thresh, dat.shape[0], 1) # compute constant offsets for thresh crossing by augmenting the ones for spike pow since both the features are related
    constant_offset = np.concatenate([constant_offset_for_spikes, constant_offset_for_thresh], axis=1, dtype='float32') # concatenate both offsets
                                       
    # cumulative noise - random walk noise
    mu = 0.0
    std = 0.02 
    cum_noise = np.cumsum(np.random.normal(mu, std, size=dat.shape[0]), dtype='float32')
    cum_noise = np.matlib.repmat(cum_noise[:,np.newaxis], 1, dat.shape[1])    

    # add all noise to the data
    dat = dat+ gaussian_noise + constant_offset + cum_noise

    # smooth augmented data
    '''
    dat = causal_smoothing_offline(dat.copy(), kernel_type = param['smoothing_kernel'], 
                                    sigma = param['smoothing_sigma'], 
                                    kernel_len=int(param['smoothing_win_len']/2)) # do slightly less smoothing to keep these trials a bit more noisy
    '''

    
    return dat


def get_mean_std(dat, dtype='float64'):
    '''
    Get mean and standard deviation of timeseries data
    dat: samples x channels
    '''
    dat_mean = np.mean(dat,axis=0) # mean along samples
    dat_std  = np.std(dat,axis=0)
    
    dat_mean = np.array(dat_mean, dtype=dtype) # float64 is required for writing to json
    dat_std  = np.array(dat_std, dtype=dtype)
    
    return dat_mean, dat_std


@jit(nopython=True)
def order_channels_using_mapping(dat, electrode_mapping):
    '''
    Reorder channels according to participant specific electrode mapping

    Args:
        dat         : neural data window [samples x channels]
        electrode_mapping:list of electrode mapping (1-indexed, like MATLAB)
    Returns:
        ordered_channels: returns ordered channels [samples x channels]
    '''

    electrode_mapping = electrode_mapping -1  # subtract 1 to make it 0-indexed
    
    #ordered_channels = dat[:,electrode_mapping] # This was the wrong channel order used on the data before Feb 25 2024 (keeping here for reference)

    ordered_channels = np.empty_like(dat)
    ordered_channels[:,electrode_mapping] = dat # This is the correct channel order used on the data after Feb 25 2024

    return ordered_channels


def undo_wrong_electrode_mapping(dat_wrong_order, electrode_mapping):
    '''
    The NSP outputs neural data in a "scrambled" order, i.e. it is not ordered by array or electrode.
    To make it easier to perform array or electrode analyses, online we electrode map (or "unscramble")
    the neural data according to a participant specific electrode mapping.

    Neural data before Feb 25 2024 underwent a wrong electrode mapping during feature extraction, so
    we now we have `undo_wrong_electrode_mapping` to reverse that wrong electrode mapping to the original "scrambled"
    order that was received from the NSP.
    After using `undo_wrong_electrode_mapping` to do that, you may then apply the correct electrode mapping function:
        
        order_channels_using_mapping(dat, electrode_mapping)
    
    to get the neural data in correctly-mapped electrode order.
    
    If you are applying `undo_wrong_electrode_mapping` to 512 features (e.g., threshold crossings + spike bandpower),
    apply it separately to each feature (256 channels) and then concatenate them.

    `undo_wrong_electrode_mapping` is *only* for applying to the data recorded before Feb 25 2024. It is not
    part of the online real-time pipeline for new data.
    
    In addition to checking the date, you can find in the .rdb the key `using_correct_electrode_mapping`
    with value `1` to confirm that data used the correct electrode mapping.

    Args:
        dat_wrong_order   : neural data with wrong electrode mapping [samples x channels]
        electrode_mapping : list of electrode mapping (1-indexed, like MATLAB)
    Returns:
        original_scrambled: returns original scrambled ordered channels [samples x channels]
    '''
    electrode_mapping = electrode_mapping -1  # subtract 1 to make it 0-indexed

    original_scrambled = np.empty_like(dat_wrong_order)
    original_scrambled[:,electrode_mapping] = dat_wrong_order
    
    return original_scrambled


# ------------------------------------------------------------------------------------
# Wrapper functions to emulate continuous data emitting for offline processing
# ------------------------------------------------------------------------------------

def wrapper_to_emulate_continuous_emitting(dat, win_size, shift_size, func, arg):
    '''
    Wrapper function to emulate continuous data emitting by sliding windows. This function also calls a function
    given in 'func' to process each data window. The last argument is a single argument taken by 'func'. 
    Currently, 'func' can be following functions from feature_extraction_helpers:
    get_spike_bandpower with argument clip_thresh (typically 10000)
    bin_data with argument bin_type (0 for average, 1 for sum)
    get_threshold_crossings with argument threshold 
    Args:
        dat       : Continuous neural stream (samples x channels)
        win_size  : Size of the window 
        shift_size: Size of the shift between windows (can be same as win_size for non-overlapping windows)
        func      : Function to process each window
        arg       : Argument to be passed to the function
    Returns:
        output    : Output of the function for each window appended together (samples' x channels)
    '''

    output = []

    for i in range(0, dat.shape[0], shift_size): #dat.shape[0]-win_size
        
        dat_1ms = dat[i:i+win_size,:]
        
        single_window_output = func(dat_1ms, arg)
        
        output.append(single_window_output)

    return np.array(output)


def filter_continuous_data_by_windowing(dat, b, a, fs, num_win_filtering, pack_per_call=1):
    '''
    Wrapper function to filter continuous data by windowing to emulate online filtering in featureExtraction node
    Args:
        dat              : Continuous neural stream (samples x channels)
        b, a             : filter coefficients
        fs               : sampling frequency
        num_win_filtering: number of 1ms windows to filter at a time including historic data
        pack_per_call    : number of 1ms windows in each call to filter at a time
    Returns:
        filt_1ms         : Filtered data (samples x channels) of type int16
    '''
    n_samples           = int(fs/1000)     # number of samples in each 1ms window
    n_channels          = dat.shape[1]     # number of channels
    filt_1ms            = np.zeros(shape=dat.shape, dtype='int16')  # array to store filtered data
    dat_len             = n_samples*pack_per_call # when pack_per_call is 1, this is 1ms window

    # initialise neural buffer of size num_win_filter x packs_per_call size
    neural_buffer = np.zeros((n_samples*pack_per_call*num_win_filtering, n_channels), 
                                dtype=np.int16)  # Here for filtering with historic data, 1 window = 1 ms * packs_per_call

    # A for loop that emulates emitting n_samples*pack_per_call samples at a time
    for i in range(dat_len, dat.shape[0]-dat_len, dat_len):

        neural_data = dat[i-dat_len:i,:] # extract data window

        neural_buffer[0:-neural_data.shape[0],:] = neural_buffer[neural_data.shape[0]:,:] # shift previous windows to left 
        neural_buffer[-neural_data.shape[0]:,:] = neural_data                             # insert new data on right
        
        # Filter the neural buffer
        filt_data = filter_signal(neural_buffer.copy(), n_samples, b, a, neural_data.shape[0])

        filt_1ms[i-dat_len:i,:] = filt_data
  
    return filt_1ms


def sliding_normalization_offline(dat, sliding_norm_size_s, bin_size_ms, norm_mean, norm_std):
    '''
    Normalisation over sliding window - this function is for offline use only
    It emulates online sliding normalization 
    Args:
        dat                : binned neural data [bins x channels]
        sliding_norm_size_s:  buffer size of the past neural data to calculate norm parameters in seconds
        bin_size_ms        : bin size in milliseconds
        norm_mean          : precomputed (from previous block) normalization mean to start with
        norm_std           : precomputed (from previous block) normalization std to start with
    Returns:
        norm_dat           : bin-wise normalized data
    '''

    norm_buf_size = sliding_norm_size_s * int(1000/bin_size_ms)   # Num of bins to use for normalisation e.g., calculate num of bins in 10s window
    norm_buf = np.zeros(shape=(norm_buf_size, dat.shape[1]), dtype='float32')
    norm_dat = np.zeros(shape=dat.shape, dtype='float32')

    buf_dat_cnt = 0  # counter to keep track of how much data has been added to the buffer
    
    for i in range(dat.shape[0]):
        
        # Add data to buffer on the right by moving previous data to the left
        norm_buf[:-1,:] = norm_buf[1:,:]
        norm_buf[-1, :] = dat[i,:]
        
        # if buffer is full with previous data, compute new normalization parameters, else old parameters will be used
        if buf_dat_cnt > norm_buf_size:
            
            # Get mean and std from last sliding_norm_size_s length data buffer
            norm_mean, norm_std = get_mean_std(norm_buf)
            
        buf_dat_cnt +=1
        
        # Normalise using appropriate mean and std 
        norm_dat[i,:] = zscore(dat[i,:], norm_mean, norm_std)

    norm_dat = norm_dat.astype('float32')

    return norm_dat


def causal_smoothing_offline(dat, kernel_type = 'half_gaussian', sigma = 3.0, kernel_len=15):
    '''
    Causal smoothing by weighted average of past values using a kernal
    Args:
        dat        : Binned normalized neural data [bins x channels]
        kernel_type: smoothing kernal type
        sigma      : Kernel shape parameter
        kernel_len : Kernel length in bins. This will determine how many past bins will be used for smoothing
    Returns:
        smooth_dat : Smoothed data
    '''
    
    kernel = get_kernel(kernel_type, sigma, kernel_len) # Get kernal window for this type of kernel
    
    buf_size   = kernel.shape[0] 
    buf        = np.zeros(shape=(buf_size, dat.shape[1]), dtype='float32')
    smooth_dat = np.zeros(shape=dat.shape, dtype='float32')
        
    for i in range(dat.shape[0]):
        
        # Add data to the buffer on the right by moving the past data to the left
        buf[:-1,:] = buf[1:,:]
        buf[-1,:]  = dat[i,:]

        # Causal smoothing along time - weighted sum of buffer using weights given in the kernel
        temp = np.sum(np.multiply(buf, kernel), axis=0)
        
        smooth_dat[i, :] = temp
    
    return smooth_dat


# -------------------------------------------------------------------------------------------
# Functions to read and write from json, mat, yaml, .ns5, pickle files 
# -------------------------------------------------------------------------------------------

def read_thresholds_from_json(threshold_fname):
    '''
    Function to read pre-computed thresholds from json file. 
    Args:
        threshold_fname: path to json file containing thresholds
    Returns:
        thresholds     : Returns a list
    '''

    with open(threshold_fname) as f:
        thresholds_dict = json.load(f)

    thresholds = np.array(thresholds_dict['thresholds'])
    
    return thresholds


def read_electrode_mapping(electrode_map_fname):
    '''
    Function to read electrode mapping from json file.
    Args:  
        electrode_map_fname: path to json file containing electrode mapping
    Returns:
        electrode_mapping  : Returns a list of electrode mapping (note this is 1-indexed)
    '''

    with open(electrode_map_fname) as f:
        electrode_mapping_dict = json.load(f)

    electrode_mapping = np.array(electrode_mapping_dict['electrode_mapping']) 
    
    return electrode_mapping


def read_lrr_weights(lrr_weights_fname):
    '''
    Function to read precomputed LRR weights
    Args:
        lrr_weights_fname: a .mat file containing a separate 2D weights matrix for each array
    Returns:
        lrr_weights      : Returns a list of weights matrices for each array. 
                           weights are stored in a list of length [n_arrays] with weight matrix [n_electrodes_per_array x n_electrodes_per_array]
    '''
 
    lrr_weights = scipy.io.loadmat(lrr_weights_fname)['lrr_weights']

    return lrr_weights


def read_ns5_file(ns5_filename,blackrock_util_path, n_channels, aud_channel=None):
    '''
    Read data from ns5 file
    Args:
        ns5_filename        : path to ns5 file to read
        blackrock_util_path : path to blackrock python utilities
        n_channels          : number of channels in the ns5 file
        aud_channel         : channel number of the audio channel (optional)
    Returns:
        raw_neural          : returns raw 30kHz data [samples x channels] of type int16 from the ns5 file
        raw_audio           : returns raw 30kHz audio data [samples x 1] of type int16 if aud_channel is provided
    '''

    # get blackrock python utilities to read .ns5 data
    sys.path.append(blackrock_util_path)

    from brpylib import NsxFile
    # Use version 2.1.0 of brpylib
    # ** Changed line 43 in brpylib.py file to remove '.' for relative path from '.brMiscFxns' in imports as follows:
    # line 43 -> from brMiscFxns import brmiscfxns_ver, openfilecheck 

    # Read ns5 file
    nsx_file = NsxFile(ns5_filename)

    all_dat = nsx_file.getdata('all',0) # electrode ids and start time s

    nsx_file.close()

    # Data is in the cell with maximum length, hence identify the cell containing data
    # Note that data is not always in the last cell of 'data' and can be distributed over multiple cells!! This will need to be sorted manually for now
    dat_len = 0
    dat_cell = 0
    for i in range(len(all_dat['data'])):
        if all_dat['data'][i].shape[0] > dat_len:
            dat_cell = i
            dat_len = all_dat['data'][i].shape[0]
    print('Data cell is ', dat_cell, ' Total cells are ', i)
    raw_neural = all_dat['data'][dat_cell][:,:n_channels]  # data is in last cell in 'data' and extract only first n_channel channels
    
    if aud_channel is not None:
        raw_audio  = all_dat['data'][dat_cell][:,aud_channel]  # Audio channel is usually the last channel
        return raw_neural, raw_audio
    
    return raw_neural


def read_dict_from_json(in_fname):
    '''
    Function to read dict from json. 
    Args:
        in_fname: Full path to json file to read
    Returns:
        dat_dict : Dictionary read from the json file
    '''
    with open(in_fname, 'r') as stream:
        dat_dict = json.load(stream)
  
    return dat_dict


def read_dict_from_yaml(in_fname):
    '''
    Function to read dict from yaml. 
    Args:
        in_fname: Full path to json file to read
    Returns:
        this_dict : Dictionary read from the json file
    '''
    with open(in_fname, 'r') as stream:
        dat_dict = yaml.safe_load(stream)
  
    return dat_dict


def read_from_pickle(in_fname):
    '''
    Function to read data from yaml. 
    Args:
        in_fname: Full path to pickle file to read
    Returns:
        data : data read from pickle file
    '''
    with open(in_fname, 'rb') as handle:
        dat = pickle.load(handle)

    return dat


def save_dict_in_json(out_fname, dat_dict):
    '''
    Function to save dictionary to json
    Args:
        out_fname: name with full path of output file (string)
        dat_dict : dictionary to be written
    Returns:
        out_fname: output file name
    '''
    with open(out_fname, 'w') as f:
        json.dump(dat_dict, f, indent=4)
        
    return out_fname 


def save_in_pickle(out_fname, dat):
    '''
    Function to serialize a variable and save in pickle file
    Args:
        out_fname: name with full path of output file (string)
        dat      : data to be pickled
    Returns:
        out_fname: output file name
    '''
    
    with open(out_fname, 'wb') as handle:
        pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return out_fname


def save_dict_in_mat(out_fname, dat_dict):
    '''
    Function to save dictionary to a mat file
    Args:
        out_fname: name with full path of output file (string)
        dat_dict : dictionary to be written
    Returns:
        out_fname: output file name
    '''
    # save as .mat file 
    scipy.io.savemat(out_fname, dat_dict)
    
    return out_fname