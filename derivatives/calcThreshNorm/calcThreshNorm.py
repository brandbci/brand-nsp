#! /usr/bin/env python
"""
Takes data from a dump.rdb and computes voltage thresholds
and spike rate normalization parameters,
then stores both in a file and Redis
"""

import argparse
import json
import logging
import numba
import numbers
import numpy as np
import os
import scipy.signal
import signal
import sys
import yaml
from joblib import Parallel, delayed

from brand.redis import RedisLoggingHandler

from redis import ConnectionError, Redis


###############################################
# Function definitions
###############################################

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


###############################################
# Initialize script
###############################################

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--nickname', type=str, required=True)
ap.add_argument('-i', '--redis_host', type=str, required=True)
ap.add_argument('-p', '--redis_port', type=int, required=True)
ap.add_argument('-s', '--redis_socket', type=str, required=False)
args = ap.parse_args()

NAME = args.nickname
redis_host = args.redis_host
redis_port = args.redis_port
redis_socket = args.redis_socket

loglevel = 'INFO'
numeric_level = getattr(logging, loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(format=f'[{NAME}] %(levelname)s: %(message)s',
                    level=numeric_level,
                    stream=sys.stdout)

def signal_handler(sig, frame):  # setup the clean exit code with a warning
    logging.info('SIGINT received. Exiting...')
    sys.exit(1)

# place the sigint signal handler
signal.signal(signal.SIGINT, signal_handler)


###############################################
# Connect to redis and pull supergraph
###############################################
try:
    logging.info(f"Connecting to Redis at {redis_host}:{redis_port}...")
    r = Redis(redis_host, redis_port, redis_socket, retry_on_timeout=True)
    r.ping()
except ConnectionError as e:
    logging.error(f"Error with Redis connection, check again: {e}")
    sys.exit(1)
except:
    logging.error('Failed to connect to Redis. Exiting.')
    sys.exit(1)

logging.getLogger().addHandler(RedisLoggingHandler(r, NAME))

logging.info('Redis connection successful.')

try:
    supergraph_entry = r.xrevrange(b'supergraph_stream', '+', '-', 1)[0]
except IndexError as e:
    logging.error(
        f"No model published to supergraph_stream in Redis. Exiting.")
    sys.exit(1)

entry_id, entry_dict = supergraph_entry
supergraph = json.loads(entry_dict[b'data'].decode())

graph_params = supergraph['derivatives'][NAME]['parameters']

###############################################
# Load parameters
###############################################

save_filename = r.config_get('dbfilename')['dbfilename']
save_filename = os.path.splitext(save_filename)[0]
save_filepath = r.config_get('dir')['dir']
save_filepath = os.path.dirname(save_filepath)

# which stream and key to pull data from
if 'input_stream_name' not in graph_params or 'input_stream_key' not in graph_params:
    logging.error(f'\'input_stream_name\' and \'input_stream_key\' parameters are required, exiting')
    sys.exit(1)

if len(graph_params['input_stream_name']) != len(graph_params['input_stream_key']):
    logging.error(f'There must be the same number of \'input_stream_name\'s as \'input_stream_key\'s, exiting')
    exit(1)

if not isinstance(graph_params['input_stream_name'], list):
    graph_params['input_stream_name'] = list(graph_params['input_stream_name'])

if not isinstance(graph_params['input_stream_key'], list):
    graph_params['input_stream_key'] = list(graph_params['input_stream_key'])

stream_info = []
for s, k in zip(graph_params['input_stream_name'], graph_params['input_stream_key']):
    stream_info.append({'name': s,
                        'key': k,
                        'structure': supergraph['streams'][s][k]})

# get info about amount of data in each stream
ch_per_stream = []
num_entries = []
num_samples = []
for s in stream_info:
    ch_per_stream.append(s['structure']['chan_per_stream'])
    num_entries.append(r.xlen(s['name']))
    num_samples.append(num_entries[-1]*s['structure']['samp_per_stream'])

# looks if unshuffle dict is specified and available, to enable unshuffling
if 'unshuffle_file' in graph_params:
    unshuffle_file = graph_params['unshuffle_file']
    try: 
        with open(unshuffle_file, 'r') as f:
            unshuffle_dict = json.load(f)
        logging.info(f'Array index unshuffle dict loaded from file: {unshuffle_file}')
        unshuffle = True
    except:
        logging.warning(f'Array index unshuffle dict could not be loaded from file defined by \'unshuffle_file\': {unshuffle_file}. Will assume channels are ordered correctly')
        unshuffle = False
else:
    logging.warning(f'No array index unshuffle dict provided by field \'unshuffle_file\'. Will assume channels are ordered correctly')
    unshuffle = False

# if no unshuffling, use identity matrix
if unshuffle == False:
    unshuffle_matrix = np.eye(np.sum(ch_per_stream), dtype=np.float64)
# if unshuffling, build unshuffling matrices
else:
    n_chan = np.sum(ch_per_stream)
    unshuffle_matrix = np.zeros((n_chan,n_chan), dtype=np.float64)
    electrode_mapping = np.array(unshuffle_dict['electrode_mapping'])
    for chan_in in range(n_chan):
        chan_out = electrode_mapping[chan_in]-1
        unshuffle_matrix[chan_out, chan_in] = 1

# the RMS multiplier to use to calculate voltage thresholds
if 'thresh_mult' in graph_params:
    thresh_mult = graph_params['thresh_mult']
    if not isinstance(thresh_mult, numbers.Number):
        logging.error(f'\'thresh_mult\' must be of type \'numbers.Number\', but it was {thresh_mult}. Exiting')
        sys.exit(1)
else:
    thresh_mult = -4.5

# whether to compute normalization parameters for band power features
if 'norm_bp' in graph_params:
    norm_bp = graph_params['norm_bp']
    if not isinstance(norm_bp, bool):
        logging.error(f'\'norm_bp\' must be of type \'bool\', but it was {norm_bp}. Exiting')
        sys.exit(1)
else:
    norm_bp = False

if 'ch_mask_stream' in graph_params:
    ch_mask_entry = r.xrevrange(graph_params['ch_mask_stream'], '+', '-', count=1)
    if ch_mask_entry:
        ch_mask = np.frombuffer(ch_mask_entry[0][1][b'channels'], dtype=np.uint16)
    else:
        logging.warning(f'\'ch_mask_stream\' was set to {graph_params["ch_mask_stream"]}, but there were no entries. Defaulting to using all channels')
        ch_mask = np.arange(np.sum(ch_per_stream), dtype=np.uint16)
else:
    ch_mask = np.arange(np.sum(ch_per_stream), dtype=np.uint16)

# whether to filter the data in 'input_stream_name' before calculating thresholds
if 'filter_first' in graph_params:
    filter_first = graph_params['filter_first']
    if not isinstance(filter_first, bool):
        logging.error(f'\'filter_first\' must be of type \'bool\', but it was {filter_first}. Exiting')
        sys.exit(1)
else:
    filter_first = True

# whether to causally or acausally filter
if 'causal' in graph_params:
    causal = graph_params['causal']
    if not isinstance(causal, bool):
        logging.error(f'\'causal\' must be of type \'bool\', but it was {causal}. Exiting')
        sys.exit(1)
else:
    causal = False

# filter order
if 'butter_order' in graph_params:
    butter_order = graph_params['butter_order']
    if not isinstance(butter_order, int):
        logging.error(f'\'butter_order\' must be of type \'int\', but it was {butter_order}. Exiting')
        sys.exit(1)
else:
    butter_order = 4

# filter lower cutoff
if 'butter_lowercut' in graph_params:
    butter_lowercut = graph_params['butter_lowercut']
    if not isinstance(butter_lowercut, numbers.Number) and butter_lowercut is not None:
        logging.error(f'\'butter_lowercut\' must be of type \'numbers.Number\' or \'None\', but it was {butter_lowercut}. Exiting')
        sys.exit(1)
else:
    butter_lowercut = None

# filter upper cutoff
if 'butter_uppercut' in graph_params:
    butter_uppercut = graph_params['butter_uppercut']
    if not isinstance(butter_uppercut, numbers.Number) and butter_uppercut is not None:
        logging.error(f'\'butter_uppercut\' must be of type \'numbers.Number\' or \'None\', but it was {butter_uppercut}. Exiting')
        sys.exit(1)
else:
    butter_uppercut = None

# sampling frequency
if 'samp_freq' in graph_params:
    samp_freq = graph_params['samp_freq']
    if not isinstance(samp_freq, numbers.Number) or samp_freq < 1e3:
        logging.error(f'\'samp_freq\' must be of type \'numbers.Number\' and at least 1000 Hz, but it was {samp_freq}. Exiting')
        sys.exit(1)
else:
    samp_freq = 30000

# decimation
if 'decimate' in graph_params:
    decimate = graph_params['decimate']
    if not isinstance(decimate, int) or decimate < 1:
        logging.error(f'\'decimate\' must be a positive integer, but it was {decimate}. Exiting')
        sys.exit(1)
else:
    decimate = 1

# amount of data to process in seconds
if 'data_time_s' in graph_params:
    data_time_s = graph_params['data_time_s']
    if data_time_s is not None and (not isinstance(data_time_s, numbers.Number) or data_time_s <= 0):
        logging.error(f'\'data_time_s\' must be a positive number, but it was {data_time_s}. Exiting')
        sys.exit(1)
else:
    data_time_s = None

# re-reference type
if 'rereference' in graph_params:
    reref = graph_params['rereference']
    if not isinstance(reref, str) and reref is not None:
        logging.error(f'\'rereference\' must be of type \'str\', but it was {reref}. Exiting')
        sys.exit(1)
    if isinstance(reref, str):
        if reref.lower() not in ['car', 'lrr']:
            logging.error(f'\'rereference\' must be \'CAR\' or \'LRR\', but it was {reref}. Exiting')
            sys.exit(1)
        reref = reref.lower()
elif 'enable_CAR' in graph_params:
    if not isinstance(graph_params['enable_CAR'], bool):
        logging.error(f'\'enable_CAR\' must be of type \'bool\', but it was {graph_params["enable_CAR"]}. Exiting')
        sys.exit(1)
    reref = 'car' if graph_params['enable_CAR'] else None
else:
    reref = 'car'

# list of lists of re-reference groupings
if reref is not None and ('reref_group_sizes' in graph_params or 'CAR_group_sizes' in graph_params):
    size_key = 'reref_group_sizes' if 'reref_group_sizes' in graph_params else 'CAR_group_sizes'
    reref_sizes = graph_params[size_key]
    if not isinstance(reref_sizes, list):
        if isinstance(reref_sizes, int):
            reref_sizes = []
            # get rereferencingg group sizes of the specified size, until we run out of channels for the stream
            for s in stream_info:
                ch_count = s['structure']['chan_per_stream']
                while ch_count > 0:
                    reref_sizes.append(min([graph_params[size_key], ch_count]))
                    ch_count -= graph_params[size_key]
    reref_groups = []
    ch_count = 0
    for g in reref_sizes:
        if not isinstance(g, int):
            logging.error(f'\'reref_group_sizes\' must be a list of \'int\'s or a single \'int\', but {graph_params[size_key]} was given. Exiting')
            sys.exit(1)
        reref_groups.append(np.arange(ch_count, ch_count+g).tolist())
        ch_count += g
else:
    reref_groups = []
    ch_count = 0
    for c in ch_per_stream:
        reref_groups.append(np.arange(ch_count, ch_count+c).tolist())
        ch_count += c
        
# exclude channels
if 'exclude_channels' in graph_params:
    exclude_ch = graph_params['exclude_channels']
    if not isinstance(exclude_ch, list):
        if isinstance(exclude_ch, int):
            exclude_ch = [exclude_ch]
    for c in exclude_ch:
        if not isinstance(c, int):
            logging.error(f'\'exclude_channels\' must be a list of \'int\'s or a single \'int\', but {graph_params["exclude_channels"]} was given. Exiting')
            sys.exit(1)
    for c in exclude_ch:
        for g in reref_groups:
            if c in g:
                g.remove(c)

# binning size
if 'bin_size' in graph_params:
    bin_size = graph_params['bin_size']
    if not isinstance(bin_size, int) or bin_size < 1:
        logging.error(f'\'bin_size\' must be of type \'int\' and at least 1 ms, but it was {bin_size}. Exiting')
        sys.exit(1)
else:
    bin_size = 10 # ms

# keep only masked channels
for g_idx in range(len(reref_groups)):
    reref_groups[g_idx] = list(set(reref_groups[g_idx]).intersection(set(ch_mask)))

###############################################
# Read in data
###############################################

# find how many entries to pull from each stream based on the minimum number of samples across streams
n_samples = np.min(num_samples)
if data_time_s is not None:
    data_time_samples = int(data_time_s * samp_freq)
    if n_samples < data_time_samples:
        logging.error(f'Not enough samples in data to process {data_time_s} seconds (only {n_samples} samples available, need {data_time_samples}), exiting')
        sys.exit(1)
    n_samples = data_time_samples
for idx, s in enumerate(stream_info):
    num_entries[idx] = int(np.ceil(n_samples/s['structure']['samp_per_stream']))

if n_samples == 0:
    logging.info(f'{num_samples} samples found in {graph_params["input_stream_name"]} streams, exiting')
    sys.exit(0)

# preallocate data array
all_data = np.empty((np.sum(ch_per_stream), n_samples), dtype=np.float64)

tot_ch = 0
for s, n_entries, n_ch in zip(stream_info, num_entries, ch_per_stream):
    this_ch = np.arange(tot_ch, tot_ch+n_ch)

    entries = r.xrevrange(
        s['name'],
        '+',
        '-',
        count=n_entries)

    i_start = 0
    stream_data = np.empty((n_ch, s['structure']['samp_per_stream']*n_entries), s['structure']['sample_type'])
    for _, entry_data in entries[::-1]:  # put it all into an array
        i_end = i_start + s['structure']['samp_per_stream']
        stream_data[:, i_start:i_end] = np.reshape(
            np.frombuffer(entry_data[s['key'].encode()], dtype=s['structure']['sample_type']),
            (n_ch, s['structure']['samp_per_stream']))
        i_start = i_end
    all_data[this_ch, :] = np.float64(stream_data[:, :n_samples])
    tot_ch += n_ch

logging.info(f'Processing {all_data.shape[1]} samples of data')

# unshuffle data
all_data = unshuffle_matrix @ all_data


###############################################
# Filter if requested
###############################################

if filter_first:
    if butter_lowercut and butter_uppercut:
        filt_type = 'bandpass'
        Wn = [butter_lowercut, butter_uppercut]
    elif butter_uppercut:
        filt_type = 'lowpass'
        Wn = butter_uppercut
    elif butter_lowercut:
        filt_type = 'highpass'
        Wn = butter_lowercut
    else:
        logging.error(f'Either butter low cutoff or high cutoff must be defined. Exiting')
        sys.exit(1)        

    sos = scipy.signal.butter(  butter_order,
                                Wn,
                                btype=filt_type,
                                analog=False,
                                output='sos',
                                fs=samp_freq) # set up a filter

    # initialize the state of the filter
    zi_flat = scipy.signal.sosfilt_zi(sos)
    # so that we have the right number of dimensions
    zi = np.zeros((zi_flat.shape[0], np.sum(ch_per_stream), zi_flat.shape[1]))
    # filter initialization
    for ii in range(np.sum(ch_per_stream)):
        zi[:, ii, :] = zi_flat

    # log the filter info
    causal_str = 'causal' if causal else 'acausal'
    message = (f'Using {butter_order :d} order, '
                f'{Wn} hz {filt_type} {causal_str} filter')
    message += f' with {reref.upper()}' if reref is not None else ''
    logging.info(message)

    if causal:
        all_data = scipy.signal.sosfilt(sos,
                                                         all_data,
                                                         axis=1,
                                                         zi=zi)
    else:
        all_data = scipy.signal.sosfiltfilt(sos, all_data, axis=1)

logging.debug('Finished filtering')
###############################################
# Compute rereferencing parameters
###############################################

@numba.jit('float64[:,:](float64[:,:], float64[:,:])', nopython=True)
def rereference_data(data, reref_params):
    reref_mat = np.eye(reref_params.shape[0]) - reref_params
    reref_data = reref_mat @ data
    return reref_data


def calc_lrr_params_parallel(channel, group, decimate=1):
    """
    Calculate parameters for linear regression reference. This version is
    made to be used with multiprocessing.

    Parameters
    ----------
    channel : int
        Index of the channel for which the reference is being computed
    group : array-like of shape (n_group_channels,)
        List of channels to use for the referencing
    decimate : int, optional
        Factor by which to decimate the data, by default 1 (no decimation)

    Returns
    -------
    channel : int
        Index of the channel for which the reference was computed
    group : array-like of shape (n_ref_channels,)
        List of channels used for the referencing
    params : numpy.ndarray of shape (n_ref_channels,)
        Weights of each channel to use when rereferencing
    """
    grp = np.setdiff1d(group, channel)

    X = all_data[grp, ::decimate].T
    y = all_data[channel, ::decimate].reshape(1, -1)
    params = np.linalg.solve(X.T @ X, X.T @ y.T).T

    return channel, grp, params


reref_params = np.zeros((tot_ch, tot_ch), dtype=np.float64)
if reref == 'car':
    ch_count = 0
    for g, s in zip(reref_groups, reref_sizes):
        reref_params[ch_count:ch_count+s, g] = 1./len(g)
        ch_count += s

elif reref == 'lrr':
    # use single-precision for faster compute
    all_data = all_data.astype(np.float32)
    with Parallel(n_jobs=-1, require='sharedmem') as parallel:
        # loop through the groups and compute LRR for each one
        ch_count = 0
        for idx, (g, s) in enumerate(zip(reref_groups, reref_sizes)):
            # compute the LRR parameters for each channel in this group
            tasks = [
                delayed(calc_lrr_params_parallel)(channel=ch, group=g)
                for ch in range(ch_count, ch_count + s)
            ]
            lrr_params = parallel(tasks)
            # unpack the parallel execution results - assign the LRR parameters
            # to the reref_params array
            for item in lrr_params:
                ch, grp, output = item
                reref_params[ch, grp] = output

            ch_count += s
    all_data = all_data.astype(np.float64)
# Re-reference the data
all_data = rereference_data(all_data, reref_params)
logging.debug('Finished rereferencing')

###############################################
# Compute thresholds
###############################################

thresholds = (thresh_mult *
                np.sqrt(np.mean(np.square(all_data), axis=1))).reshape(
                    -1, 1)


###############################################
# Compute normalization parameters
###############################################

crossings = ((all_data[:, :-1] > thresholds) & (all_data[:, 1:] <= thresholds))

def bin_data(data, bin_width):
    n_samples, n_chans = data.shape
    # samples that fit evenly into a bin
    b_samples = bin_width * (n_samples // bin_width)
    # bin the data
    binned_data = data[:b_samples, :].reshape(-1, bin_width,
                                              n_chans).sum(axis=1)
    return binned_data

# accumulate in 1 ms bins, then in bin_size ms bins
binned = bin_data(crossings.T, int(samp_freq/1e3)) > 0
binned = bin_data(binned, bin_size)

if norm_bp:
    binned_bp = bin_data(all_data[:, 1:].T ** 2, int(samp_freq * bin_size / 1e3)) / (samp_freq/1e3) # equivalent to bpExtraction node, divide by 30
    binned = np.hstack((binned, binned_bp))

def exponential_moving_average(data, alpha):
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = (1 - alpha) * data[i] + (alpha) * ema[i - 1]
    return ema

alpha = graph_params.get('ema_alpha', 0)
binned = exponential_moving_average(binned, alpha=alpha)

# calculate means and STDs
means = binned.mean(axis=0)
stds = binned.std(axis=0)

logging.debug('Finished computing thresholds')
###############################################
# Save file & write to Redis
###############################################

r.xadd('thresholds', {
    'thresholds': thresholds.tobytes()})

r.xadd('normalization_parameters', {
    'means': means.tobytes(),
    'stds': stds.tobytes()})

r.xadd('rereference_parameters', {
    'channel_scaling': reref_params.tobytes(),
    'channel_unshuffling': unshuffle_matrix.tobytes()})

output_dict = {
    'stream_info': stream_info,
    'thresh_mult': thresh_mult,
    'filter_first': filter_first}
if filter_first:
    output_dict['causal'] = causal
    output_dict['butter_order'] = butter_order
    output_dict['butter_passband'] = Wn
    output_dict['samp_freq'] = samp_freq
output_dict['rereference'] = reref
if reref:
    output_dict['reref_groups'] = reref_groups
output_dict['bin_size'] = bin_size
output_dict['thresholds'] = thresholds.reshape(-1).tolist()
output_dict['means'] = means.tolist()
output_dict['stds'] = stds.tolist()
output_dict['rereference_parameters'] = reref_params.tolist()
output_dict['channel_unshuffling'] = unshuffle_matrix.tolist()

save_filename = save_filename + '.yaml'
save_filepath = os.path.join(save_filepath, 'thresh_norm')
if not os.path.exists(save_filepath):
    os.makedirs(save_filepath)

save_path = os.path.join(save_filepath, save_filename)

# save the file
logging.info(f'Saving thresh_norm file to {save_path}')
with open(save_path, 'w') as f:
    yaml.dump(output_dict, f, sort_keys=False, default_flow_style=None)
