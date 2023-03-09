
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
# Load data from RDBs 

SAVE_DIR = '/samba/data/sim/2023-03-08/RawData'
RDB_DIR = os.path.join(SAVE_DIR,'RDB')
RDB_FILENAMES = {'OFF': 'sim_230308_018.rdb', 'ON': 'sim_230308_020.rdb'} 
REDIS_IP = '127.0.0.1'
REDIS_PORT = 18000

decoded_streams = {}

for adaptive_on, RDB_FILENAME in RDB_FILENAMES.items():

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

    decoded_streams[adaptive_on] = {}

    # Load replayed RDB id

    latest_supergraph_entry = r.xrevrange(b'supergraph_stream', count=1)
    supergraph = json.loads(latest_supergraph_entry[0][1][b'data'])
    replayed_rdb_path = supergraph['nodes']['replay_streams']['parameters']['rdb_file']
    replayed_block_id = os.path.basename(replayed_rdb_path).split('.')[0]

    if b'thresh_cross_1' in r.keys('*'):
        stream_data = r.xrange(b'thresh_cross_1')
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
                elif key.decode() == 'crossings':
                    dat = np.frombuffer(val, dtype=np.uint16)
                    entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
            out[i] = entry_dec
        decoded_streams[adaptive_on]['thresh_cross_1'] = out

    if b'thresh_cross_2' in r.keys('*'):
        stream_data = r.xrange(b'thresh_cross_2')
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
                elif key.decode() == 'crossings':
                    dat = np.frombuffer(val, dtype=np.uint16)
                    entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
            out[i] = entry_dec
        decoded_streams[adaptive_on]['thresh_cross_2'] = out

    # Shutdown Redis

    r.shutdown(nosave=True)

# %%
# Convert to pandas dataframe

thresh_cross_1_on_df = pd.DataFrame(decoded_streams['ON']['thresh_cross_1'])
thresh_cross_2_on_df = pd.DataFrame(decoded_streams['ON']['thresh_cross_2'])
thresh_cross_1_on = np.vstack(thresh_cross_1_on_df['crossings']).T
thresh_cross_2_on = np.vstack(thresh_cross_2_on_df['crossings']).T
max_idx = np.min([thresh_cross_1_on.shape[1], thresh_cross_2_on.shape[1]])
thresh_cross_on = np.vstack((thresh_cross_1_on[:,:max_idx], thresh_cross_2_on[:,:max_idx])).astype(np.int)

thresh_cross_1_off_df = pd.DataFrame(decoded_streams['OFF']['thresh_cross_1'])
thresh_cross_2_off_df = pd.DataFrame(decoded_streams['OFF']['thresh_cross_2'])
thresh_cross_1_off = np.vstack(thresh_cross_1_off_df['crossings']).T
thresh_cross_2_off = np.vstack(thresh_cross_2_off_df['crossings']).T
max_idx = np.min([thresh_cross_1_off.shape[1], thresh_cross_2_off.shape[1]])
thresh_cross_off = np.vstack((thresh_cross_1_off[:,:max_idx], thresh_cross_2_off[:,:max_idx])).astype(np.int)

# %% 

# Plot threshold crossings

SAMPLES_PLOT = 10000

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 12), constrained_layout=True, facecolor='white')

#axs[0].plot(thresh_cross_1[0:,:SAMPLES_PLOT].T)
axs[0].matshow(thresh_cross_off[0:,:SAMPLES_PLOT], aspect='auto')
axs[0].axvline(1000, color='w')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Channel #')
axs[0].set_title('Adaptive thresholds OFF')

axs[1].matshow(thresh_cross_on[0:,:SAMPLES_PLOT], aspect='auto')
axs[1].axvline(1000, color='w')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Channel #')
axs[1].set_title('Adaptive thresholds ON')

plt.suptitle(f"Spikes for replayed block: {replayed_block_id}", fontsize=16)

plt.show()


# %% 

# Plot difference in threshold crossings

SAMPLES_PLOT = 1000*2*10

fig, ax  = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10, 12), constrained_layout=True, facecolor='white')

diff_crossings = thresh_cross_on[0:,:SAMPLES_PLOT] - thresh_cross_off[0:,:SAMPLES_PLOT]

#axs[0].plot(thresh_cross_1[0:,:SAMPLES_PLOT].T)
ax.matshow(diff_crossings+1, aspect='auto', interpolation='nearest')
ax.axvline(1000, color='w')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Channel #')

plt.suptitle(f"Spikes differences after turning on adaptive thresholds for replayed block: {replayed_block_id}", fontsize=16)

plt.show()

# %%
