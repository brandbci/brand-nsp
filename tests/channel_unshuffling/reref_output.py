# %%

import numpy as np
import matplotlib.pyplot as plt
import redis

# %%

r = redis.Redis(host='192.168.30.6', port=27263)

# %%

entries = r.xrevrange(b'nsp_neural', count=1)

sample = np.frombuffer(entries[0][1][b'samples'],dtype=np.int16).reshape((256, 30))

print(sample.shape)

# %%

fig = plt.figure(figsize=(10,10), facecolor='w', constrained_layout=True)
plt.matshow(sample)
plt.colorbar()
plt.xlabel('sample time number')
plt.ylabel('electrode #')
plt.title('re-reference input', fontsize=24)
plt.savefig('./input.png', bbox_inches="tight")

# %%

entries = r.xrevrange(b'reref_neural', count=1)

sample = np.frombuffer(entries[0][1][b'samples'],dtype=np.float32).reshape((256, 30))

print(sample.shape)

# %%

fig = plt.figure(figsize=(10,10), facecolor='w', constrained_layout=True)
plt.matshow(sample)
plt.colorbar()
plt.xlabel('sample time number')
plt.ylabel('electrode #')
plt.title('re-reference output', fontsize=24)
plt.savefig('./output.png', bbox_inches="tight")


# %%
