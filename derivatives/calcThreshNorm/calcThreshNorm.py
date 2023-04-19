#! /usr/bin/env python
"""
Takes data from a dump.rdb and computes voltage thresholds
and spike rate normalization parameters,
then stores both in a file and Redis
"""

import json
import logging
import numbers
import numpy as np
import os
import scipy.signal
import signal
import sys
import yaml

from brand.redis import xread_count
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

NAME = 'calcThreshNorm'

rdb_file = sys.argv[1]

redis_host = sys.argv[2]
redis_port = sys.argv[3]

save_filename = os.path.splitext(rdb_file)[0]
save_filepath = sys.argv[4]

# set up logging
loglevel = 'INFO'
numeric_level = getattr(logging, loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(format=f'[{NAME}] %(levelname)s: %(message)s',
                    level=numeric_level,
                    stream=sys.stdout)


###############################################
## setting up clean exit code
###############################################
def signal_handler(sig, frame):  # setup the clean exit code with a warning
    logging.info('SIGINT received. Exiting...')
    sys.exit(0)

# place the sigint signal handler
signal.signal(signal.SIGINT, signal_handler)


###############################################
# Connect to redis and pull supergraph
###############################################
try:
    logging.info(f"Connecting to Redis at {redis_host}:{redis_port}...")
    r = Redis(redis_host, redis_port, retry_on_timeout=True)
    r.ping()
except ConnectionError as e:
    logging.error(f"Error with Redis connection, check again: {e}")
    sys.exit(1)
except:
    logging.error('Failed to connect to Redis. Exiting.')
    sys.exit(1)

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

# the RMS multiplier to use to calculate voltage thresholds
if 'thresh_mult' in graph_params:
    thresh_mult = graph_params['thresh_mult']
    if not isinstance(thresh_mult, numbers.Number):
        logging.error(f'\'thresh_mult\' must be of type \'numbers.Number\', but it was {thresh_mult}. Exiting')
        sys.exit(1)
else:
    thresh_mult = -4.5

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
else:
    reref = None

# list of lists of re-reference groupings
if reref is not None and 'reref_group_sizes' in graph_params:
    reref_sizes = graph_params['reref_group_sizes']
    if not isinstance(reref_sizes, list):
        if isinstance(reref_sizes, int):
            reref_sizes = []
            # get rereferencingg group sizes of the specified size, until we run out of channels for the stream
            for s in stream_info:
                ch_count = s['structure']['chan_per_stream']
                while ch_count > 0:
                    reref_sizes.append(min([graph_params['reref_group_sizes'], ch_count]))
                    ch_count -= graph_params['reref_group_sizes']
    reref_groups = []
    ch_count = 0
    for g in reref_sizes:
        if not isinstance(g, int):
            logging.error(f'\'reref_group_sizes\' must be a list of \'int\'s or a single \'int\', but {graph_params["reref_group_sizes"]} was given. Exiting')
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
for idx, s in enumerate(stream_info):
    num_entries[idx] = int(np.ceil(n_samples/s['structure']['samp_per_stream']))

if n_samples == 0:
    logging.info(f'{num_samples} samples found in {graph_params["input_stream_name"]} streams, exiting')
    sys.exit(0)

# preallocate data array
all_data = np.empty((np.sum(ch_per_stream), n_samples), dtype=np.float64)

logging.info(f'Computing thresholds from {n_samples} samples')

tot_ch = 0
for s, n_entries, n_ch in zip(stream_info, num_entries, ch_per_stream):
    this_ch = np.arange(tot_ch, tot_ch+n_ch)

    reply = xread_count(r,
                        stream=s['name'],
                        startid='0',
                        count=n_entries,
                        block=0)

    _, entries = reply[0]  # get the list of entries

    i_start = 0
    stream_data = np.empty((n_ch, s['structure']['samp_per_stream']*n_entries), s['structure']['sample_type'])
    for _, entry_data in entries:  # put it all into an array
        i_end = i_start + s['structure']['samp_per_stream']
        stream_data[:, i_start:i_end] = np.reshape(
            np.frombuffer(entry_data[s['key'].encode()], dtype=s['structure']['sample_type']),
            (n_ch, s['structure']['samp_per_stream']))
        i_start = i_end
    all_data[this_ch, :] = np.float64(stream_data[:, :n_samples])
    tot_ch += n_ch


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


###############################################
# Compute rereferencing parameters
###############################################

reref_params = np.zeros((tot_ch, tot_ch), dtype=np.float64)
if reref == 'car':
    ch_count = 0
    for g, s in zip(reref_groups, reref_sizes):
        reref_params[ch_count:ch_count+s, g] = 1./len(g)
        ch_count += s

elif reref == 'lrr':
    ch_count = 0
    for g, s in zip(reref_groups, reref_sizes):
        for ch in range(ch_count, ch_count+s):
            grp = np.setdiff1d(g, ch)
            X = all_data[grp, :].T
            y = all_data[ch, :].reshape(1, -1)
            reref_params[ch, grp] = (y @ X) @ np.linalg.inv(X.T @ X) # sklearn is slow
        ch_count += s

reref_mat = -reref_params
np.fill_diagonal(reref_mat, 1. + np.diag(reref_mat))

all_data = reref_mat @ all_data


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

# calculate means and STDs
means = binned.mean(axis=0)
stds = binned.std(axis=0)


###############################################
# Save file & write to Redis
###############################################

r.xadd('thresholds', {
    'thresholds': thresholds.tobytes()})

r.xadd('normalization_parameters', {
    'means': means.tobytes(),
    'stds': stds.tobytes()})

r.xadd('rereference_parameters', {
    'channel_scaling': reref_params.tobytes()})

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
output_dict['rereference_parameters'] = reref_params.tolist()
output_dict['thresholds'] = thresholds.reshape(-1).tolist()
output_dict['means'] = means.tolist()
output_dict['stds'] = stds.tolist()

save_filename = save_filename + '.yaml'
save_filepath = os.path.join(save_filepath, 'thresh_norm')
if not os.path.exists(save_filepath):
    os.makedirs(save_filepath)

save_path = os.path.join(save_filepath, save_filename)

# save the file
logging.info(f'Saving thresh_norm file to {save_path}')
with open(save_path, 'w') as f:
    yaml.dump(output_dict, f, sort_keys=False, default_flow_style=None)