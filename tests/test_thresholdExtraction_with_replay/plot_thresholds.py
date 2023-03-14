
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

SAVE_DIR = '/samba/data/sim/2023-03-14/RawData'
RDB_DIR = os.path.join(SAVE_DIR,'RDB')
RDB_FILENAME = 'sim_230314_008.rdb'
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
# Load replayed RDB id

latest_supergraph_entry = r.xrevrange(b'supergraph_stream', count=1)
supergraph = json.loads(latest_supergraph_entry[0][1][b'data'])
replayed_rdb_path = supergraph['nodes']['replay_streams']['parameters']['rdb_file']
replayed_block_id = os.path.basename(replayed_rdb_path).split('.')[0]

# %% 
# Load RMS stream data

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
                dat = np.frombuffer(val, dtype=np.float64)
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
                dat = np.frombuffer(val, dtype=np.float64)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
            elif key.decode() == 'thresholds':
                dat = np.frombuffer(val, dtype=np.float64)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
        out[i] = entry_dec
    decoded_streams['rms_continuous_2'] = out

adaptive_threholding = False
if b'rms_continuous_1' in r.keys('*') and b'rms_continuous_2' in r.keys('*'):
    adaptive_threholding = True


# %% 
# Load threshold crossing stream data

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
    decoded_streams['thresh_cross_1'] = out

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
    decoded_streams['thresh_cross_2'] = out

thresh_cross_1_df = pd.DataFrame(decoded_streams['thresh_cross_1'])
thresh_cross_2_df = pd.DataFrame(decoded_streams['thresh_cross_2'])

thresh_cross_1 = np.vstack(thresh_cross_1_df['crossings']).T
thresh_cross_2 = np.vstack(thresh_cross_2_df['crossings']).T

# %% 
# Load threshold crossing stream data

if b'binned_spikes' in r.keys('*'):
    stream_data = r.xrange(b'binned_spikes')
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
                dat = np.frombuffer(val, dtype=np.int8)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
        out[i] = entry_dec
    decoded_streams['binned_spikes'] = out

binned_spikes_df = pd.DataFrame(decoded_streams['binned_spikes'])

# %%
# Load ch_mask stream data

if b'z_mask_stream' in r.keys('*'):
    stream_data = r.xrevrange(b'z_mask_stream')
    ch_mask = np.frombuffer(stream_data[0][1][b'channels'], dtype=np.uint16)

# %% 
# Plot adaptive threshold results

if adaptive_threholding:

    rms_1_df = pd.DataFrame(decoded_streams['rms_continuous_1'])
    rms_2_df = pd.DataFrame(decoded_streams['rms_continuous_2'])

    rms_1 = np.vstack(rms_1_df['samples']).T
    rms_2 = np.vstack(rms_2_df['samples']).T
    thresholds_1 = np.vstack(rms_1_df['thresholds']).T
    thresholds_2 = np.vstack(rms_2_df['thresholds']).T

    # Plot RMS values

    SAMPLES_PLOT = 200000

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 10), constrained_layout=True, facecolor='white')

    axs[0].plot(rms_1[0:,:SAMPLES_PLOT].T)
    axs[0].axvline(2000, color='k')
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('RMS continuous neural (bits?)')
    axs[0].set_title('Array 1')

    axs[1].plot(rms_2[0:,:SAMPLES_PLOT].T)
    axs[1].axvline(2000, color='k')
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('RMS continuous neural (bits?)')
    axs[1].set_title('Array 2')

    plt.suptitle(f"Continuous RMS values for replayed block: {replayed_block_id}", fontsize=16)

    plt.show()

    # Plot RMS values separated

    SAMPLES_PLOT = 200000

    # manually remove channels
    ch_mask = np.delete(ch_mask, np.where(ch_mask == 145))
    ch_mask = np.delete(ch_mask, np.where(ch_mask == 160))
    # array specific channel masks
    ch_mask_1 = np.isin(np.arange(0,192), ch_mask)[:96]
    ch_mask_2 = np.isin(np.arange(0,192), ch_mask)[96:]

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 10), constrained_layout=True, facecolor='white')

    axs[0,0].plot(rms_1[ch_mask_1,:SAMPLES_PLOT].T)
    axs[0,0].axvline(2000, color='k')
    axs[0,0].set_xlabel('Time (ms)')
    axs[0,0].set_ylabel('RMS continuous neural (bits?)')
    axs[0,0].set_title('Array 1 (masked-in channels)')

    axs[0,1].plot(rms_1[~ch_mask_1,:SAMPLES_PLOT].T)
    axs[0,1].axvline(2000, color='k')
    axs[0,1].set_xlabel('Time (ms)')
    axs[0,1].set_ylabel('RMS continuous neural (bits?)')
    axs[0,1].set_title('Array 1 (masked-out channels)')

    axs[1,0].plot(rms_2[ch_mask_2,:SAMPLES_PLOT].T)
    axs[1,0].axvline(2000, color='k')
    axs[1,0].set_xlabel('Time (ms)')
    axs[1,0].set_ylabel('RMS continuous neural (bits?)')
    axs[1,0].set_title('Array 2 (masked-in channels)')

    axs[1,1].plot(rms_2[~ch_mask_2,:SAMPLES_PLOT].T)
    axs[1,1].axvline(2000, color='k')
    axs[1,1].set_xlabel('Time (ms)')
    axs[1,1].set_ylabel('RMS continuous neural (bits?)')
    axs[1,1].set_title('Array 2 (masked-out channels)')

    plt.suptitle(f"Continuous RMS values for replayed block: {replayed_block_id}", fontsize=16)

    plt.show()

    # Plot thresholds 

    SAMPLES_PLOT = 10000

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 10), constrained_layout=True, facecolor='white')

    axs[0].plot(thresholds_1[0:,:SAMPLES_PLOT].T)
    axs[0].axvline(2000, color='k')
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Spike thresholds (bits?)')
    axs[0].set_title('Array 1')

    axs[1].plot(thresholds_2[0:,:SAMPLES_PLOT].T)
    axs[1].axvline(2000, color='k')
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Spike thresholds (bits?)')
    axs[1].set_title('Array 2')

    plt.suptitle(f"Adaptive thresholds for replayed block: {replayed_block_id}", fontsize=16)

    plt.show()

# %% 
# Plot threshold crossings

SAMPLES_PLOT = 10000

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 10), constrained_layout=True, facecolor='white')

axs[0].matshow(thresh_cross_1[0:,:SAMPLES_PLOT], aspect='auto')
axs[0].axvline(2000, color='k')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Channel #')
axs[0].set_title('Array 1')

axs[1].matshow(thresh_cross_2[0:,:SAMPLES_PLOT], aspect='auto')
axs[1].axvline(2000, color='k')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Channel #')
axs[1].set_title('Array 2')

plt.suptitle(f"Spikes for replayed block: {replayed_block_id}, adaptive thresholds: {b'rms_continuous_1' in r.keys('*')}", fontsize=16)

plt.show()

# %%
# Analyze timing of threshold crossings node
tc1_df_timing = thresh_cross_1_df.set_index('nsp_idx_1')
bs_timing = binned_spikes_df.set_index('nsp_idx_1')

timing_df = tc1_df_timing.join(bs_timing, how='outer', lsuffix='_tc1', rsuffix='_bs')
timing_df.fillna(method="ffill", inplace=True)
timing_df.dropna(inplace=True)

init_sample = 1
timing_df = timing_df[init_sample:-1:10]

#timing_df.fillna(method="bfill", inplace=True)
procesing_time = (timing_df['ts_bs'] - timing_df['ts_tc1']) * 1e-6

fig = plt.subplots(figsize=(10, 10), constrained_layout=True, facecolor='white')

plt.hist(procesing_time.values)

plt.show()

# %% 
# Shutdown Redis

r = redis.Redis(host=REDIS_IP, port=REDIS_PORT)
r.shutdown(nosave=True)
# %%
