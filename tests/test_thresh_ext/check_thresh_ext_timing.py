# %%
import redis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

r = redis.Redis()

# %%
replies = r.xrange('thresholdCrossings')
data = [None] * len(replies)
for i, (entry_id, entry_data) in enumerate(replies):
    index = np.frombuffer(entry_data[b'timestamps'], dtype=np.uint32)[0]
    timestamp = np.frombuffer(entry_data[b'BRANDS_time'], dtype=np.uint64)[0]
    data[i] = {'i_tc': index, 'ts_tc': timestamp}
tc_df = pd.DataFrame(data)

# %%
replies = r.xrange('thresh_ext_tester')
data = [None] * len(replies)
for i, (entry_id, entry_data) in enumerate(replies):
    index = np.frombuffer(entry_data[b'timestamps'], dtype=np.uint32)[0]
    timestamp = np.frombuffer(entry_data[b't'], dtype=np.uint64)[0]
    data[i] = {'i_ca': index, 'ts_ca': timestamp}
ca_df = pd.DataFrame(data)

# %%
graph_df = tc_df.set_index('i_tc',
                           drop=False).join(ca_df.set_index('i_ca',
                                                            drop=False))

# %%
# threshold extraction latency
latency = (graph_df['ts_tc'] - graph_df['ts_ca']) / 1e6
print(f'threshold extraction latency: {latency.mean()} +- {latency.std()}'
      f' ({latency.min()} - {latency.max()})')

# %%
# inter-sample interval
print(f"threshold extraction ISI: {graph_df['ts_tc'].diff().mean() / 1e6}")
print(f"input ISI: {graph_df['ts_ca'].diff().mean() / 1e6}")

# %%
bin_width = 0.01  # ms
plt.hist(latency,
         bins=np.arange(0, latency.max() + bin_width, step=bin_width),
         histtype='step')
plt.yscale('log')
plt.xlabel('Latency (ms)')

# %%
plt.scatter(latency.reset_index().index, latency, marker='.')
plt.xlabel('Sample Index')
plt.ylabel('Latency (ms)')

# %%
