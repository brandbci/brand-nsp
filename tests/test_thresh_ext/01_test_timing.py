# %%
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import redis
import yaml

test_dir = os.path.dirname(__file__)
using_ipython = hasattr(sys, 'ps1')

graph_path = os.path.join(test_dir, 'test_thresh_ext.yaml')
with open(graph_path, 'r') as f:
    graph = yaml.safe_load(f)

# %%
# connect to Reds
r = redis.Redis()

# %%
# Start graph
print(f'Starting graph from {graph_path} as JSON')
last_id = r.xadd('supervisor_ipstream', {
    'commands': 'startGraph',
    'graph': json.dumps(graph)
})

# %%
# wait for the graph to stop
done = False
while not done:
    streams = r.xread({'supervisor_ipstream': last_id}, block=0)
    key, messages = streams[0]
    last_id, data = messages[0]
    cmd = (data[b'commands']).decode("utf-8")
    if cmd == "stopGraph":
        done = True

# %%
# Load stream entries
replies = r.xrange('thresholdCrossings')
data = [None] * len(replies)
for i, (entry_id, entry_data) in enumerate(replies):
    index = np.frombuffer(entry_data[b'timestamps'], dtype=np.uint32)[0]
    timestamp = np.frombuffer(entry_data[b'BRANDS_time'], dtype=np.uint64)[0]
    data[i] = {'i_tc': index, 'ts_tc': timestamp}
tc_df = pd.DataFrame(data)

replies = r.xrange('thresh_ext_tester')
data = [None] * len(replies)
for i, (entry_id, entry_data) in enumerate(replies):
    index = np.frombuffer(entry_data[b'timestamps'], dtype=np.uint32)[0]
    timestamp = np.frombuffer(entry_data[b't'], dtype=np.uint64)[0]
    data[i] = {'i_ca': index, 'ts_ca': timestamp}
ca_df = pd.DataFrame(data)

graph_df = tc_df.set_index('i_tc',
                           drop=False).join(ca_df.set_index('i_ca',
                                                            drop=False))

# %%
# analyze timing
# threshold extraction latency
latency = (graph_df['ts_tc'] - graph_df['ts_ca']) / 1e6
print(f'threshold extraction latency: {latency.mean()} +- {latency.std()}'
      f' ({latency.min()} - {latency.max()})')

# inter-sample interval
print(f"threshold extraction ISI: {graph_df['ts_tc'].diff().mean() / 1e6}")
print(f"input ISI: {graph_df['ts_ca'].diff().mean() / 1e6}")

# %%
# save data
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

date_str = datetime.now().strftime(r'%y%m%dT%H%M')
filepath = os.path.join(data_dir, f'{date_str}_test_thresh_ext.pkl')
with open(filepath, 'wb') as f:
    pickle.dump(graph_df, f)
print(f'data saved to {filepath}')

# %%
# Delete streams from this test
new_streams = ['thresholdCrossings', 'thresholds', 'thresh_ext_tester']
for stream in new_streams:
    r.delete(stream)
r.memory_purge()
print(f'Deleted streams: {new_streams}')

# %%
