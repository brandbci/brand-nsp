#! /usr/bin/env python
"""
Writes array based channel masks to Redis
"""

import argparse
import glob
import json
import logging
import numbers
import numpy as np
import os
import re
import signal
import sys
import yaml

from brand.redis import RedisLoggingHandler

from datetime import datetime

from dotenv import dotenv_values

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
    logging.info('No arrays specified, assuming all arrays are included')
    arrays = array_names

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

###############################################
# Build channel mask using the specified arrays
###############################################

array_channel_boundaries = np.cumsum(array_sizes)
channel_mask = []
arrays_used_in_order = []
for array, size, boundary in zip(array_names, array_sizes, array_channel_boundaries):
    if array in arrays:
        channel_mask.append(np.arange(boundary - size, boundary))
        arrays_used_in_order.append(array)

channel_mask = np.concatenate(channel_mask)

# combine with external channel masks
for ext_mask in ext_channel_masks:
    channel_mask = np.intersect1d(channel_mask, ext_mask)

channel_mask = np.sort(channel_mask)

logging.info(f'Electrodes being included: {channel_mask.tolist()}')

###############################################
# Store channel mask in Redis
###############################################

mask_stream = 'array_mask'
logging.info(f'Storing channel mask in stream {mask_stream}')
r.xadd(mask_stream,
       {'channels': channel_mask.astype(np.uint16).tobytes()})