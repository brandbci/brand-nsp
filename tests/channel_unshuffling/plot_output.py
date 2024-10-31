
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

participant = 'sim256'
session = '2024-01-05'
block = f'{participant}_240105_000'
path = f'/samba/data/{participant}/{session}/RawData/thresh_norm/{block}.yaml'

with open(path, 'r') as file:
    yaml_dict = yaml.safe_load(file)

reref_param = np.array(yaml_dict['rereference_parameters'])
plt.matshow(reref_param)
plt.colorbar()
plt.xlabel('NSP channel #')
plt.ylabel('electrode #')
plt.savefig('./reref_param.png')

unshuffle_matrix = np.array(yaml_dict['channel_unshuffling'])
plt.matshow(unshuffle_matrix)
plt.colorbar()
plt.xlabel('NSP channel #')
plt.ylabel('electrode #')
plt.savefig('./unshuffle_matrix.png')

reref_matrix = (np.eye(reref_param.shape[0]) - reref_param) @ unshuffle_matrix
plt.matshow(reref_matrix)
plt.colorbar()
plt.xlabel('NSP channel #')
plt.ylabel('electrode #')

plt.savefig('./reref_matrix.png')

print(reref_matrix.min())
print(reref_matrix.max())

