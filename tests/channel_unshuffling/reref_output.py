# %%

import numpy as np
import matplotlib.pyplot as plt
import redis

# %%

r = redis.Redis()

# %%

entries = r.xrevrange(b'nsp_neural', count=1)

sample = np.frombuffer(entries[0][1][b'samples'],dtype=np.int16).reshape((256, 30))

print(sample.shape)

# %%

fig = plt.figure(facecolor='w')
plt.matshow(sample)
plt.colorbar()
plt.xlabel('sample time number')
plt.ylabel('electrode #')
plt.show()

# %%

entries = r.xrevrange(b'reref_neural', count=1)

sample = np.frombuffer(entries[0][1][b'samples'],dtype=np.float32).reshape((256, 30))

print(sample.shape)

# %%

fig = plt.figure(facecolor='r')
plt.matshow(sample)
plt.colorbar()
plt.xlabel('sample time number')
plt.ylabel('electrode #')
plt.show()


# %%
