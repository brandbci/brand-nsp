#! /usr/bin/env python
"""
Takes data from a dump.rdb and computes voltage thresholds
and spike rate normalization parameters,
then stores both in a file and Redis
"""

from brand.redis import xread_count
import json
import logging
import numbers
import numpy as np
import os
from redis import ConnectionError, Redis
import scipy.signal
import signal
import sys
import yaml

###############################################
# Initialize script
###############################################

NAME = 'updateThresholds'

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
        "No model published to supergraph_stream in Redis. Exiting.")
    sys.exit(1)

entry_id, entry_dict = supergraph_entry
supergraph = json.loads(entry_dict[b'data'].decode())

graph_params = supergraph['derivatives'][NAME]['parameters']

###############################################
# Load parameters
###############################################

# which stream and key to pull data from
if 'input_stream_name' not in graph_params or 'input_stream_key' not in graph_params or 'nsp_channels' not in graph_params:
    logging.error('\'input_stream_name\', \'input_stream_key\', and \'nsp_channels\' parameters are required, exiting')
    sys.exit(1)

# which stream and key to pull data from
if 'thresholds_stream' not in graph_params:
    logging.error('\'thresholds_stream\' parameter is required, exiting')
    sys.exit(1)

if (len(graph_params['input_stream_name']) != len(graph_params['input_stream_key']) and
        len(graph_params['input_stream_name']) != len(graph_params['nsp_channels'])):
    logging.error('There must be the same number of \'input_stream_name\'s , \'input_stream_key\'s, and \'nsp_channels\', exiting')
    exit(1)

if not isinstance(graph_params['input_stream_name'], list):
    graph_params['input_stream_name'] = list(graph_params['input_stream_name'])

if not isinstance(graph_params['input_stream_key'], list):
    graph_params['input_stream_key'] = list(graph_params['input_stream_key'])

if not isinstance(graph_params['nsp_channels'], list):
    graph_params['nsp_channels'] = list(graph_params['nsp_channels'])

###############################################
# Read in data
###############################################

# preallocate data array
all_data = np.empty(np.sum(graph_params['nsp_channels']), dtype=np.float64)

tot_ch = 0
for i, stream in enumerate(graph_params['input_stream_name']):
    latest_entry = r.xrevrange(stream.encode(), count=1)
    latest_data = np.frombuffer(latest_entry[0][1][graph_params['input_stream_key'][i].encode()],
                                dtype=np.float64)
    ch = graph_params['nsp_channels'][i]
    all_data[tot_ch:tot_ch+ch] = np.float64(latest_data)
    tot_ch += ch

###############################################
# Write to Redis
###############################################

r.xadd(graph_params['thresholds_stream'], {
    'thresholds': all_data.tobytes()})

thresholds_stream = graph_params['thresholds_stream']
logging.info(f'Logging latest adaptive thresholds to stream: \'{thresholds_stream}\'')