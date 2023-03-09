
# %% 
# Imports

import os
import sys
import time
import redis
import yaml
import json
import logging
import coloredlogs
import subprocess
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')
logger.setLevel(logging.DEBUG)

# %% 
# Start Redis 

SAVE_DIR = '/samba/data/sim/2023-03-08/RawData'
RDB_DIR = os.path.join(SAVE_DIR,'RDB')
RDB_FILENAME = 'sim_230308_020.rdb'
REDIS_IP = '127.0.0.1'
REDIS_PORT = 18000

redis_command = ['redis-server', '--bind', REDIS_IP, '--port', str(REDIS_PORT)]
redis_command.append('--dbfilename')
redis_command.append(RDB_FILENAME)
redis_command.append('--dir')
redis_command.append(RDB_DIR)

print('Starting redis: ' + ' '.join(redis_command))

proc = subprocess.Popen(redis_command, stdout=subprocess.PIPE)
redis_pid = proc.pid

try:
    out, _ = proc.communicate(timeout=1)
    if out:
        print(out.decode())
    if 'Address already in use' in str(out):
        print("Could not run redis-server (address already in use). Check if a Redis server is already running on that port. Aborting.")
        exit(1)
    else:
        print("Launching redis-server failed for an unknown reason, check supervisor logs. Aborting.")
        exit(1)
except subprocess.TimeoutExpired:  # no error message received
    print('Redis-server is running.')

r = redis.Redis(host=REDIS_IP, port=REDIS_PORT)

busy_loading = True 
while busy_loading:
    try:
        print(f"Streams in database: {r.keys('*')}")
        busy_loading = False
    except redis.exceptions.BusyLoadingError:
        print('Redis is busy loading dataset in memory')
        time.sleep(1)

# %% 
# Load stream data

decoded_streams  = {}

if b'rms_continuous_1' in r.keys('*'):
    stream_data = r.xrange(b'rms_continuous_1')
    out = [None] * len(stream_data)
    for i, (entry_id, entry_data) in tqdm(enumerate(stream_data)):
        entry_dec = {}
        for key, val in entry_data.items():
            if key.decode() == 'ts':
                dat = np.frombuffer(val, dtype=np.uint64)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
            elif key.decode() == 'sync':
                sync_dict = json.loads(val)
                for sync_key, sync_val in sync_dict.items():
                    entry_dec[sync_key] = sync_val
            elif key.decode() == 'samples':
                dat = np.frombuffer(val, dtype=np.float32)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
            elif key.decode() == 'thresholds':
                dat = np.frombuffer(val, dtype=np.float64)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
        out[i] = entry_dec
    decoded_streams['rms_continuous_1'] = out
if b'rms_continuous_2' in r.keys('*'):
    stream_data = r.xrange(b'rms_continuous_2')
    out = [None] * len(stream_data)
    for i, (entry_id, entry_data) in tqdm(enumerate(stream_data)):
        entry_dec = {}
        for key, val in entry_data.items():
            if key.decode() == 'ts':
                dat = np.frombuffer(val, dtype=np.uint64)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
            elif key.decode() == 'sync':
                sync_dict = json.loads(val)
                for sync_key, sync_val in sync_dict.items():
                    entry_dec[sync_key] = sync_val
            elif key.decode() == 'samples':
                dat = np.frombuffer(val, dtype=np.float32)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
            elif key.decode() == 'thresholds':
                dat = np.frombuffer(val, dtype=np.float64)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
        out[i] = entry_dec
    decoded_streams['rms_continuous_2'] = out

rms_1_df = pd.DataFrame(decoded_streams['rms_continuous_1'])
rms_2_df = pd.DataFrame(decoded_streams['rms_continuous_2'])

rms_1 = np.vstack(rms_1_df['samples']).T
rms_2 = np.vstack(rms_2_df['samples']).T
thresholds_1 = np.vstack(rms_1_df['thresholds']).T
thresholds_2 = np.vstack(rms_2_df['thresholds']).T

# %%
# Load replayed RDB id

latest_supergraph_entry = r.xrevrange(b'supergraph_stream', count=1)
supergraph = json.loads(latest_supergraph_entry[0][1][b'data'])
replayed_rdb_path = supergraph['nodes']['replay_streams']['parameters']['rdb_file']
replayed_block_id = os.path.basename(replayed_rdb_path).split('.')[0]

# if b'thresh_cross_1' in r.keys('*'):
#     stream_data = r.xrange(b'thresh_cross_1')
#     for i, (entry_id, entry_data) in tqdm(enumerate(stream_data)):
#         entry_dec = {}
#         for key, val in entry_data.items():
#             if key.decode() in spec:
#                 dtype = spec[key.decode()]
#                 if dtype == 'str':
#                     entry_dec[key.decode()] = val.decode()
#                 elif dtype == 'sync':
#                     sync_dict = json.loads(val)
#                     for sync_key, sync_val in sync_dict.items():
#                         entry_dec[sync_key] = sync_val
#                 elif dtype == 'timeval':
#                     entry_dec[key.decode()] = np.array(
#                         timevals_to_timestamps(val)) * 1e9
#                 elif dtype == 'timespec':
#                     entry_dec[key.decode()] = np.array(
#                         timespecs_to_timestamps(val)) * 1e9
#                 else:
#                     dat = np.frombuffer(val, dtype=dtype)
#                     entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
#         out[i] = entry_dec
#     decoded_streams[stream.decode()] = out

# %% 
# Plot results

# Plot RMS values

SAMPLES_PLOT = 200000

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 10), constrained_layout=True)

axs[0].plot(rms_1[0:,:SAMPLES_PLOT].T)
axs[0].axvline(1000, color='k')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('RMS continuous neural (bits?)')
axs[0].set_title('Array 1')

axs[1].plot(rms_2[0:,:SAMPLES_PLOT].T)
axs[1].axvline(1000, color='k')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('RMS continuous neural (bits?)')
axs[1].set_title('Array 2')

plt.suptitle(f"Continuous RMS values for replayed block: {replayed_block_id}", fontsize=16)

plt.show()

# Plot thresholds 

SAMPLES_PLOT = 3000

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 10), constrained_layout=True)

axs[0].plot(thresholds_1[0:,:SAMPLES_PLOT].T)
axs[0].axvline(1000, color='k')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Spike thresholds (bits?)')
axs[0].set_title('Array 1')

axs[1].plot(thresholds_2[0:,:SAMPLES_PLOT].T)
axs[1].axvline(1000, color='k')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Spike thresholds (bits?)')
axs[1].set_title('Array 2')

plt.suptitle(f"Adaptive thresholds for replayed block: {replayed_block_id}", fontsize=16)

plt.show()

# %% 
# Shutdown Redis

r = redis.Redis(host=REDIS_IP, port=REDIS_PORT)
r.shutdown(nosave=True)

