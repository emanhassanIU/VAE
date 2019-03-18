import os
import random
import numpy as np
#-------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
#-------------------------------------
dset_dir = '/share/jproject/fg508/eman/domainAdapt/GANStyle/VAE_tut/dsprites-dataset/' 
dataset_zip = np.load(dset_dir+'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

print('Keys in the dataset:', dataset_zip.keys())
imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
print('latent values size ')
print(latents_values.shape)
latents_classes = dataset_zip['latents_classes']
print(' latents_classes ')
print(latents_classes.shape)
metadata = dataset_zip['metadata'][()]

print(' data size values')
print(latents_values[0,:])

print(' dta size 2 classes')
print(latents_classes[0,:])

#print('Metadata: \n', metadata)



#root = os.path.join(dset_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
#data = np.load(root, encoding='latin1')
#print(' data 1 ')
#data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
#print(' data 2 ')
#print(data.shape)
#train_kwargs = {'data_tensor':data}
#print(' DONE ')
