# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

# %%
data_info = [
    ['data/221011T1605_test_thresh_ext.pkl', 'acausal IIR-FIR'],
    ['data/221011T1620_test_thresh_ext.pkl', 'acausal IIR-IIR'],
    ['data/221011T1640_test_thresh_ext.pkl', 'causal IIR'],
]
graph_dfs = [None] * len(data_info)
for i, (path, _) in enumerate(data_info):
    with open(path, 'rb') as f:
        graph_dfs[i] = pickle.load(f)

run_ids = [os.path.basename(path).split('_')[0] for path, _ in data_info]

# %%
fig_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(fig_dir, exist_ok=True)

fig, axes = plt.subplots(ncols=1,
                         nrows=1,
                         figsize=(9, 3),
                         sharex='row',
                         sharey='row')
bin_width = 0.01  # ms

for i, graph_df in enumerate(graph_dfs):
    graph_df = graph_df[graph_df.index != 0]
    latency = (graph_df['ts_tc'] - graph_df['ts_ca']) / 1e6
    if i in [0, 1]:
        latency -= 4
    axes.hist(latency,
              bins=np.arange(0, latency.max() + bin_width, step=bin_width),
              histtype='step',
              label=data_info[i][1])
    axes.set_yscale('log')
    axes.set_xlabel('Latency (ms)')
plt.legend()
plt.tight_layout()
run_str = '_'.join(run_ids)
plt.savefig(os.path.join(fig_dir, f'{run_str}_latency_hist.pdf'))

# %%
fig, axes = plt.subplots(ncols=1,
                         nrows=len(graph_dfs),
                         figsize=(9, 3 * len(graph_dfs)),
                         sharex=True,
                         sharey=True)
for i, graph_df in enumerate(graph_dfs):
    graph_df = graph_df[graph_df.index != 0]
    latency = (graph_df['ts_tc'] - graph_df['ts_ca']) / 1e6
    if i in [0, 1]:
        latency -= 4
    filter_label = data_info[i][1]
    title = (f'{filter_label} latency: {latency.mean():.8f} +- '
             f'{latency.std():.8f}'
             f' ({latency.min():.8f} - {latency.max():.8f}) ms')
    axes[i].set_title(title)

    axes[i].scatter(latency.reset_index().index,
                    latency,
                    marker='.',
                    color=f'C{i}')
    axes[i].set_xlabel('Sample Index')
    axes[i].set_ylabel('Latency (ms)')

    plt.tight_layout()
run_str = '_'.join(run_ids)
plt.savefig(os.path.join(fig_dir, f'{run_str}_latency_scatter.png'))
# %%
