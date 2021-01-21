import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import pandas as pd
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transform
from torchvision.utils import save_image
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import creatdir, get_filepaths, SPP_make, SPP_rec, read_wav, data_generator, valid_generator
from model import Enhance, Speaker

from scipy.io import wavfile
import scipy.io
import librosa
import soundfile as sf
import numpy.matlib
from sklearn import manifold
import seaborn as sns

from scipy.io import wavfile

#a = np.load('new_dataset_lstm1out_mask_noisetype_ACC.npy')

os.environ["CUDA_VISIBLE_DEVICES"]="2" #Your GPU number, default = 0

random.seed(999)

batch_size= 1
Epoch = 20

######################### Testing data #########################
clean_path = "..." # write your own validation clean data path here

Test_Noisy_paths = np.load('val_new_Noisy.npy') # read the noisy data from val_new_Noisy.npy (have been shuffle already)

speaker_label = pd.read_csv('TMHINT_new_valid_label.csv') # You need to write your own clean speaker label

############ Read models ##################### 
model_enh = Enhance().cuda()
model_enh.load_state_dict(torch.load('...')) # read your own enhance model
model_enh.eval()

model_spk = Speaker().cuda()
model_spk.load_state_dict(torch.load('...')) # read your own speaker model
model_spk.eval()

n = np.zeros((1,4))
count = np.zeros((1,4))
acc = []
spk_list = []
label_list = []

########## Using validation data to computing speaker accuracy ####################
print ('testing...')
start_time = time.time()
for iter in tqdm(range(len(Test_Noisy_paths))):
    n_sp , c_sp , frames_label = data_generator(Test_Noisy_paths, clean_path, iter, speaker_label)

    S=Test_Noisy_paths[iter].split('/')
    dB=S[-3]
    wave_name=S[-1]
    noise_name=S[-2]

    enh_out, lstm2_out = model_enh(torch.from_numpy(n_sp.astype('float32')).cuda() , None)

    lstm2_out = lstm2_out.cpu().detach().numpy()
    lstm2_out = lstm2_out.squeeze()
    lstm2_out = np.pad(lstm2_out,((5,5),(0,0)),'reflect')
    lstm2_out_con = np.zeros([624,3300]).astype('float32')
    for i in range(624):
        lstm2_out_con[i,:] = np.concatenate((lstm2_out[i+5-5,:],lstm2_out[i+5-4,:],lstm2_out[i+5-3,:],lstm2_out[i+5-2,:],lstm2_out[i+5-1,:],lstm2_out[i+5,:],lstm2_out[i+5+1,:],lstm2_out[i+5+2,:],lstm2_out[i+5+3,:],lstm2_out[i+5+4,:],lstm2_out[i+5+5,:]),axis=0)

    speaker_out, dense3_out = model_spk(torch.from_numpy(lstm2_out_con.astype('float32')).cuda())
 

    running_acc_initial = 0.0
    prediction = torch.argmax(speaker_out, dim=1)
    running_acc_initial += torch.sum(prediction == torch.from_numpy(frames_label).cuda())

    # Computing speaker accuracy in different noise environment 
    if noise_name == 'engine':
        n[0,0] += running_acc_initial/len(frames_label)
        count[0,0] += 1
    elif noise_name == 'PINKNOISE_16k':
        n[0,1] += running_acc_initial/len(frames_label)
        count[0,1] += 1
    elif noise_name == 'new_street':
        n[0,2] += running_acc_initial/len(frames_label)
        count[0,2] += 1
    elif noise_name == 'white':
        n[0,3] += running_acc_initial/len(frames_label)
        count[0,3] += 1

    spk_list.append(dense3_out.cpu().detach())
    label_list.append(frames_label)

for i in range(len(n[0,:])):
    acc.append(n[0,i] / count[0,i])
np.save('new_dataset_lstm1out_mask_noisetype_ACC.npy', np.array(acc))

####### Doing t-sne on our speaker embeddings ############################
x = np.concatenate(spk_list[0:51])
y = np.concatenate(label_list[0:51])

tsne = manifold.TSNE(n_components=2, init='random', random_state=501)
x_tsne = tsne.fit_transform(torch.from_numpy(x))

x_min, x_max = x_tsne.min(0), x_tsne.max(0)
x_norm = (x_tsne - x_min) / (x_max - x_min)

palette = sns.color_palette("bright", 7)
sns_plot = sns.scatterplot(x_norm[:,0], x_norm[:,1], hue=y, legend = False, palette=palette, s=10)
fig = sns_plot.get_figure()
fig.savefig('lstm1out_mask_t-SNE.png')


end_time = time.time()
print ('The testing for this file ran for %.2fm' % ((end_time - start_time) / 60.))
