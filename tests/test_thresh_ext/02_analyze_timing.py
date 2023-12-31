# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

# %%
data_path = 'data/221011T1620_test_thresh_ext.pkl'
with open(data_path, 'rb') as f:
    graph_df = pickle.load(f)
run_id = os.path.basename(data_path).split('_')[0]

# %%
fig_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# %%
# threshold extraction latency
graph_df = graph_df[graph_df.index != 0]
latency = (graph_df['ts_tc'] - graph_df['ts_ca']) / 1e6
print(f'threshold extraction latency: {latency.mean()} +- {latency.std()}'
      f' ({latency.min()} - {latency.max()})')
plt.plot(latency, '.')
# %%
# inter-sample interval
print(f"threshold extraction ISI: {graph_df['ts_tc'].diff().mean() / 1e6}")
print(f"input ISI: {graph_df['ts_ca'].diff().mean() / 1e6}")

# %%
fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(9, 6))
bin_width = 0.01  # ms
axes[0].hist(latency,
             bins=np.arange(0, latency.max() + bin_width, step=bin_width),
             histtype='step')
axes[0].set_yscale('log')
axes[0].set_xlabel('Latency (ms)')
axes[0].set_title(f'Latency: {latency.mean():.8f} +- {latency.std():.8f}'
                  f' ({latency.min():.8f} - {latency.max():.8f}) ms')

axes[1].scatter(latency.reset_index().index, latency, marker='.')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Latency (ms)')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{run_id}_latency.png'))

# %%
