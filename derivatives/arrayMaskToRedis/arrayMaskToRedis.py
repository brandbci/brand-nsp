#! /usr/bin/env python
"""
Writes array based channel masks to Redis
"""

import argparse
import json
import logging
import numpy as np
import os
import signal
import sys

from brand.redis import RedisLoggingHandler

from dotenv import dotenv_values

from itertools import combinations

from redis import ConnectionError, Redis


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

graph_params:dict = supergraph['derivatives'][NAME]['parameters']

###############################################
# Load parameters
###############################################

# get Redis-related parameters
save_filepath = r.config_get('dir')['dir']
save_filepath = os.path.dirname(save_filepath)

# get environment variables
env_filepath = os.path.join(save_filepath, 'environment.sh')
env = dotenv_values(env_filepath)

# load array names
array_names = env['ARRAY_NAMES']
array_names = array_names.replace('(', '').replace(')', '').replace('"', '')
array_names = array_names.split(' ')

# load total number of channels from the session environment file
array_sizes = env['ARRAY_SIZES']
array_sizes = array_sizes.replace('(', '').replace(')', '')
array_sizes = array_sizes.split(' ')
array_sizes = [int(s) for s in array_sizes]

# which arrays to include in the mask
if 'arrays' in graph_params and graph_params['arrays'] is not None and graph_params['arrays']:
    arrays = graph_params['arrays']
    if not isinstance(arrays, str):
        logging.error('arrays must be a comma-separated list of strings, exiting')
        sys.exit(1)
    arrays = arrays.split(',')
    for array in arrays.copy():
        if array not in array_names:
            logging.warning(f'Array {array} not found in session environment file, skipping')
            arrays.remove(array)

    if not arrays:
        logging.error('No valid arrays specified, exiting')
        sys.exit(1)
else:
    logging.info('No arrays specified')
    arrays = None

# combine with another channel mask, if desired
ext_channel_masks = []
if 'channel_mask_stream' in graph_params and graph_params['channel_mask_stream'] is not None and graph_params['channel_mask_stream']:
    channel_mask_stream = graph_params['channel_mask_stream']
    if isinstance(channel_mask_stream, str):
        channel_mask_stream = [channel_mask_stream]
    if not all(isinstance(stream, str) for stream in channel_mask_stream):
        logging.error('Channel mask streams must be strings, exiting')
        sys.exit(1)
    
    for stream in channel_mask_stream:
        channel_mask_entry = r.xrevrange(stream, '+', '-', count=1)
        if channel_mask_entry:
            ext_channel_masks.append(np.frombuffer(channel_mask_entry[0][1][b'channels'], dtype=np.uint16))
            logging.info(f'Loaded channel mask from {channel_mask_stream}')
        else:
            logging.warning(f'Could not find channel mask in stream {channel_mask_stream}, skipping')
    
    logging.info('This derivative assumes channel masks are in the unshuffled order (i.e. electrodes belonging to each array are contiguous)')

# check for stream prefix parameter
if 'stream_prefix' in graph_params and graph_params['stream_prefix'] is not None:
    stream_prefix = graph_params['stream_prefix']
    if not isinstance(stream_prefix, str):
        logging.error('stream_prefix must be a string, exiting')
        sys.exit(1)
else:
    stream_prefix = 'session:'

###############################################
# Build  masks using all array combinations
###############################################

# get all possible array combinations
array_combinations = [c
                      for num_arrays in range(1, len(array_names) + 1)
                      for c in combinations(range(len(array_names)), num_arrays)]

array_channel_boundaries = np.cumsum(array_sizes)
channel_mask = {comb: [] for comb in array_combinations}
for comb in channel_mask:
    for array_idx in comb:
        channel_mask[comb].append(
            np.arange(
                array_channel_boundaries[array_idx] - array_sizes[array_idx],
                array_channel_boundaries[array_idx]))
    channel_mask[comb] = np.concatenate(channel_mask[comb])

    # combine with external channel masks
    for ext_mask in ext_channel_masks:
        channel_mask[comb] = np.intersect1d(channel_mask[comb], ext_mask)

    channel_mask[comb] = np.sort(channel_mask[comb])

###############################################
# Store channel mask in Redis
###############################################

# default mask stream if specified
mask_stream = 'array_mask'

# write all masks to Redis
p = r.pipeline()
for comb in channel_mask:
    stream_name = '_'.join([array_names[i] for i in comb])
    stream_name = f'{stream_prefix}{mask_stream}_{stream_name}'
    p.xadd(stream_name, {'channels': channel_mask[comb].astype(np.uint16).tobytes()})
if arrays is not None:
    arrays_idx = [array_names.index(array) for array in arrays]
    arrays_idx.sort()
    logging.info(f'Electrodes being included in array_mask stream: {channel_mask[tuple(arrays_idx)].tolist()}')
    p.xadd(mask_stream,
        {'channels': channel_mask[tuple(arrays_idx)].astype(np.uint16).tobytes()})
p.execute()

logging.info('Array masks written to Redis.')