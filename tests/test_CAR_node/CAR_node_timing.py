
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
from brand import timing

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')
logger.setLevel(logging.DEBUG)

# %% 
# Start Redis 

SAVE_DIR = '/home/samnt/Projects/emory-cart/Data/sim/2023-04-19/RawData'
RDB_DIR = os.path.join(SAVE_DIR,'RDB')
RDB_FILENAME = 'sim_230419T2013_test_CAR_node.rdb'
REDIS_IP = '127.0.0.1'
REDIS_PORT = 37105

redis_command = ['/home/samnt/Projects/emory-cart/brand/bin/redis-server', '--bind', REDIS_IP, '--port', str(REDIS_PORT)]
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

N_per_array = 96

decoded_streams = {}

if b'nsp_neural_1' in r.keys('*'):
    stream_data = r.xrange(b'nsp_neural_1')
    out = [None] * len(stream_data)
    for i, (entry_id, entry_data) in tqdm(enumerate(stream_data)):
        entry_dec = {}
        for key, val in entry_data.items():
            if key.decode() == 'timestamps':
                dat = np.frombuffer(val, dtype=np.uint32)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
            elif key.decode() == 'BRANDS_time':
                entry_dec[key.decode()] = np.array(
                    timing.timespecs_to_timestamps(val)) 
            elif key.decode() == 'samples':
                dat = np.frombuffer(val, dtype=np.int16).reshape(N_per_array, -1)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
        out[i] = entry_dec
    decoded_streams['nsp_neural_1'] = out

if b'nsp_neural_2' in r.keys('*'):
    stream_data = r.xrange(b'nsp_neural_2')
    out = [None] * len(stream_data)
    for i, (entry_id, entry_data) in tqdm(enumerate(stream_data)):
        entry_dec = {}
        for key, val in entry_data.items():
            if key.decode() == 'timestamps':
                dat = np.frombuffer(val, dtype=np.uint32)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
            elif key.decode() == 'BRANDS_time':
                entry_dec[key.decode()] = np.array(
                    timing.timespecs_to_timestamps(val)) 
            elif key.decode() == 'samples':
                dat = np.frombuffer(val, dtype=np.int16).reshape(N_per_array, -1)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
        out[i] = entry_dec
    decoded_streams['nsp_neural_2'] = out

nsp_neural_1_df = pd.DataFrame(decoded_streams['nsp_neural_1'])
nsp_neural_2_df = pd.DataFrame(decoded_streams['nsp_neural_2'])

nsp_neural_1_df['nsp_idx'] = nsp_neural_1_df.apply(lambda x: x['timestamps'][0], axis=1)
nsp_neural_2_df['nsp_idx'] = nsp_neural_2_df.apply(lambda x: x['timestamps'][0], axis=1)
nsp_neural_1_df['ts'] = nsp_neural_1_df.apply(lambda x: x['BRANDS_time'][0], axis=1)
nsp_neural_2_df['ts'] = nsp_neural_2_df.apply(lambda x: x['BRANDS_time'][0], axis=1)

nsp_neural_1_df.set_index('nsp_idx', inplace=True)
nsp_neural_2_df.set_index('nsp_idx', inplace=True)

if b'reref_neural_1' in r.keys('*'):
    stream_data = r.xrange(b'reref_neural_1')
    out = [None] * len(stream_data)
    for i, (entry_id, entry_data) in tqdm(enumerate(stream_data)):
        entry_dec = {}
        for key, val in entry_data.items():
            if key.decode() == 'timestamps':
                dat = np.frombuffer(val, dtype=np.uint32)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
            elif key.decode() == 'BRANDS_time':
                entry_dec[key.decode()] = np.array(
                    timing.timespecs_to_timestamps(val)) 
            elif key.decode() == 'samples':
                dat = np.frombuffer(val, dtype=np.float64).reshape(N_per_array, -1)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
        out[i] = entry_dec
    decoded_streams['reref_neural_1'] = out

if b'reref_neural_2' in r.keys('*'):
    stream_data = r.xrange(b'reref_neural_2')
    out = [None] * len(stream_data)
    for i, (entry_id, entry_data) in tqdm(enumerate(stream_data)):
        entry_dec = {}
        for key, val in entry_data.items():
            if key.decode() == 'timestamps':
                dat = np.frombuffer(val, dtype=np.uint32)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
            elif key.decode() == 'BRANDS_time':
                entry_dec[key.decode()] = np.array(
                    timing.timespecs_to_timestamps(val)) 
            elif key.decode() == 'samples':
                dat = np.frombuffer(val, dtype=np.float64).reshape(N_per_array, -1)
                entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
        out[i] = entry_dec
    decoded_streams['reref_neural_2'] = out

if b'rereference_parameters' in r.keys('*'):
    stream_data = r.xrange(b'rereference_parameters')
    rereference_parameters = np.frombuffer(stream_data[-1][1][b'channel_scaling'], dtype=np.float64).reshape(N_per_array*2, -1)

reref_neural_1_df = pd.DataFrame(decoded_streams['reref_neural_1'])
reref_neural_2_df = pd.DataFrame(decoded_streams['reref_neural_2'])

reref_neural_1_df['nsp_idx'] = reref_neural_1_df.apply(lambda x: x['timestamps'][0], axis=1)
reref_neural_2_df['nsp_idx'] = reref_neural_2_df.apply(lambda x: x['timestamps'][0], axis=1)
reref_neural_1_df['ts'] = reref_neural_1_df.apply(lambda x: x['BRANDS_time'][0], axis=1)
reref_neural_2_df['ts'] = reref_neural_2_df.apply(lambda x: x['BRANDS_time'][0], axis=1)

reref_neural_1_df.set_index('nsp_idx', inplace=True)
reref_neural_2_df.set_index('nsp_idx', inplace=True)

if b'thresh_cross_1' in r.keys('*'):
    stream_data = r.xrange(b'thresh_cross_1')
    out = [None] * len(stream_data)
    for i, (entry_id, entry_data) in tqdm(enumerate(stream_data)):
        entry_dec = {}
        for key, val in entry_data.items():
            if key.decode() == 'ts':
                dat = np.frombuffer(val, dtype=np.uint64).astype(np.float64) * 1e-9
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
                dat = np.frombuffer(val, dtype=np.uint64).astype(np.float64) * 1e-9
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

thresh_cross_1_df.set_index('nsp_idx_1', inplace=True)
thresh_cross_2_df.set_index('nsp_idx_2', inplace=True)

if b'binned_spikes' in r.keys('*'):
    stream_data = r.xrange(b'binned_spikes')
    out = [None] * len(stream_data)
    for i, (entry_id, entry_data) in tqdm(enumerate(stream_data)):
        entry_dec = {}
        for key, val in entry_data.items():
            if key.decode() == 'ts':
                dat = np.frombuffer(val, dtype=np.uint64).astype(np.float64) * 1e-9
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

binned_spikes_df.set_index('nsp_idx_1', inplace=True)

# %%

nsp_1_arr = np.hstack(nsp_neural_1_df['samples'].values)
reref_1_arr = np.hstack(reref_neural_1_df['samples'].values)
reref_mat = -rereference_parameters
np.fill_diagonal(reref_mat, 1. + np.diag(reref_mat))
recomputed_reref_1_arr = reref_mat[:N_per_array, :N_per_array] @ reref_1_arr

sample_chs = [0, 1, 2, 3]
T = 3000  # 30*milliseconds
t = np.arange(T)/30

_, axs = plt.subplots(len(sample_chs), 2, sharex=True, sharey=True, figsize=(10, 10), facecolor='w')

for i, ch in enumerate(sample_chs):
    axs[i, 0].plot(t, nsp_1_arr[ch, :T], color='b')
    axs[i, 1].plot(t, reref_1_arr[ch, :T], color='g')
    axs[i, 1].plot(t, recomputed_reref_1_arr[ch, :T], color='y')
    axs[i, 0].set_ylabel('ch {}'.format(ch))

axs[-1, 0].set_xlabel('Time [ms]')
axs[-1, 1].set_xlabel('Time [ms]')
axs[0, 0].set_title('cerebusAdapter')
axs[0, 1].set_title('reReference')

# plt.plot(reref_1_arr[0,:3000])
# plt.plot(nsp_1_arr[0,:3000]+1000)
plt.show()

# %%

all_df_1 = nsp_neural_1_df.join(reref_neural_1_df, how='inner', lsuffix='_nsp', rsuffix='_reref')
all_df_1 = all_df_1.join(thresh_cross_1_df, how='inner', rsuffix='_thresh')
all_df_1 = all_df_1.join(binned_spikes_df, how='inner', rsuffix='_binned')

all_df_2 = nsp_neural_2_df.join(reref_neural_2_df, how='inner', lsuffix='_nsp', rsuffix='_reref')
all_df_2 = all_df_2.join(thresh_cross_2_df, how='inner', rsuffix='_thresh')
all_df_2 = all_df_2.join(binned_spikes_df, how='inner', rsuffix='_binned')

_, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 6), facecolor='w')

axs[0].plot(1e3*(all_df_1['ts_reref']  - all_df_1['ts_nsp']), label='Re-reference')
axs[0].plot(1e3*(all_df_1['ts']        - all_df_1['ts_nsp']), label='Threshold extraction')
#axs[0].plot(1e3*(all_df_1['ts_binned'] - all_df_1['ts_nsp']), label='Binning')

axs[1].plot(1e3*(all_df_2['ts_reref']  - all_df_2['ts_nsp']), label='Re-reference')
axs[1].plot(1e3*(all_df_2['ts']        - all_df_2['ts_nsp']), label='Threshold extraction')
#axs[1].plot(1e3*(all_df_2['ts_binned'] - all_df_2['ts_nsp']), label='Binning')

axs[0].set_xlabel('NSP 1 timestamp')
axs[1].set_xlabel('NSP 2 timestamp')
axs[0].set_ylabel('Latency [ms]')
axs[1].legend()

# %% 
# Shutdown Redis

r = redis.Redis(host=REDIS_IP, port=REDIS_PORT)
r.shutdown(nosave=True)
# %%
