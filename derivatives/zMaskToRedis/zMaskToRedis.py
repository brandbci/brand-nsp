#! /usr/bin/env python
"""
Takes impedance files, computes a channel mask, and stores it in Redis
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

# impedance threshold should be a single number
if 'impedance_threshold' in graph_params:
    impedance_threshold = graph_params['impedance_threshold']
    try:
        impedance_threshold = float(impedance_threshold)
    except ValueError:
        logging.error('\'impedance_threshold\' parameter must be a number, exiting')
        sys.exit(1)

    if impedance_threshold < 0:
        logging.error('\'impedance_threshold\' parameter must be positive, exiting')
        sys.exit(1)
else:
    impedance_threshold = 800

# channels to exclude should be a single number of a comma-separated string of numbers
if 'exclude_channels' in graph_params:
    exclude_channels = graph_params['exclude_channels']
    if isinstance(exclude_channels, numbers.Number):
        exclude_channels = [exclude_channels]
    elif isinstance(exclude_channels, str):
        exclude_channels = exclude_channels.split(',')
        exclude_channels = [int(c) for c in exclude_channels if len(c) > 0]
else:
    exclude_channels = []

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

# load total number of channels from the session environment file
array_sizes = env['ARRAY_SIZES']
array_sizes = array_sizes.replace('(', '').replace(')', '')
array_sizes = array_sizes.split(' ')
array_sizes = [int(s) for s in array_sizes]

# load array names
array_names = env['ARRAY_NAMES']
array_names = array_names.replace('(', '').replace(')', '').replace('"', '')
array_names = array_names.split(' ')

###############################################
# Search for most recent impedance files
###############################################

participant_dir = os.path.dirname(os.path.dirname(save_filepath))
dirs_with_impedance = glob.glob(
    os.path.join(
        participant_dir,
        '*',
        'RawData',
        'Impedance',
        '*mpedance*')) # wheter I is capitalized or not
dirs_with_impedance.sort(reverse=True)

if len(dirs_with_impedance) == 0:
    logging.warning(f'No impedance files found in any of the date folders in {participant_dir}. Generating impedance mask assuming {sum(array_sizes)} channels.')
    impedance_files = []

else:
    # the first list item is the most recent impedance file, so get its directory
    impedance_dir = os.path.dirname(dirs_with_impedance[0])

    # warn the user if the session date is not today
    session_date = impedance_dir.split(os.sep)[-3]
    session_date = datetime.strptime(session_date, '%Y-%m-%d')
    if session_date.date() != datetime.today().date():
        logging.warning(f'Using impedance data from {session_date.date()}, not today')

    # get the impedance files
    impedance_files = glob.glob(os.path.join(impedance_dir, '*mpedance*'))

# TODO: add impedance filenames to the session environment file
# TODO: parse impedance files in order according to the session environment file

###############################################
# Compute channel mask
###############################################

if impedance_files:
    impedance_pattern = re.compile(r'(chan|elec)(\d+)\s+(\d+)\s+(kOhm|uV)')
    impedance_data = []
    for impedance_file in impedance_files:
        # load impedance file
        with open(impedance_file, 'r') as f:
            impedance_values = f.read()

        # parse using a regular expression
        impedance_values = impedance_pattern.findall(impedance_values)

        if any('elec' in line for line in impedance_values):
            logging.warning(f'Impedance file {impedance_file} contains electrode names, please re-collect the impedance file with channel numbers')

        # sort by channel number
        impedance_values = sorted([(int(channel), int(value) if unit == 'kOhm' else float("inf")) for _, channel, value, unit in impedance_values])

        # get the impedance values in order
        kohm_values = [value for _, value in impedance_values]

        # append to impedance data
        impedance_data.extend(kohm_values)

    # convert to numpy array
    impedance_data = np.array(impedance_data)

    # only keep channels with a low impedance
    channel_mask = np.argwhere(impedance_data < impedance_threshold)

else:
    channel_mask = np.arange(sum(array_sizes))

# exclude specified channels
channel_mask = np.setdiff1d(channel_mask, exclude_channels)

logging.info(f'NSP channels being included: {channel_mask.tolist()}')

if unshuffle:
    channel_mask = np.array(
        unshuffle_dict['electrode_mapping'])[channel_mask] - 1

channel_mask = np.sort(channel_mask)

logging.info(f'Electrodes being included: {channel_mask}')

###############################################
# Store channel mask in Redis
###############################################

r.xadd('z_mask_stream',
       {'channels': channel_mask.astype(np.uint16).tobytes()})