import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import os, pdb
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import scipy.io
import librosa
import soundfile as sf
import time  
import numpy as np
import numpy.matlib
import random
import torch.nn as nn
from util import creatdir, get_filepaths, SPP_make, SPP_rec, read_wav, data_generator, valid_generator
from new_dataset_lstm1out_mask_model import Enhance, Speaker

from scipy.io import wavfile

######################### Testing data #########################

Test_Noisy_paths = get_filepaths("...", '.wav') # write your own testing noisy data path here

random.shuffle(Test_Noisy_paths)

Test_Clean_paths = get_filepaths("...", '.wav') # write your own testing clean data path here
                
Num_testdata=len(Test_Noisy_paths)   

############ Read models ##################### 
model_enh = Enhance()
model_enh.load_state_dict(torch.load('...')) # read your own enhance model
model_enh.eval()

model_spk = Speaker()
model_spk.load_state_dict(torch.load('...')) # read your own speaker model
model_spk.eval()

############ Creating files for saving testing wav #################
if not os.path.exists("./new_dataset_lstm1out_mask_20epoch"):
    os.makedirs("./new_dataset_lstm1out_mask_20epoch")

SNRs = ['SNR_-5', 'SNR_0', 'SNR_5']
Noises = ['engine', 'PINKNOISE_16k', 'new_street', 'white']
for N in Noises:
    for SNR in SNRs:
        if not os.path.exists("./new_dataset_lstm1out_mask_20epoch/"+N+"/"+SNR):
            os.makedirs("./new_dataset_lstm1out_mask_20epoch/"+N+"/"+SNR)

########### Start testing ##################
print ('testing...')
start_time = time.time()
for path in Test_Noisy_paths:   
    S=path.split('/')
    dB=S[-3]
    wave_name=S[-1]
    noise_name=S[-2]

    noisy = read_wav(path)
    n_sp, n_p = SPP_make(noisy, Noisy=True)


    enh_out, lstm2_out = model_enh(torch.from_numpy(n_sp.astype('float32')),None)

    lstm2_out = lstm2_out.cpu().detach().numpy()
    lstm2_out = lstm2_out.squeeze()
    lstm2_out = np.pad(lstm2_out,((5,5),(0,0)),'reflect')
    lstm2_out_con = np.zeros([499,3300]).astype('float32')
    for i in range(499):
        lstm2_out_con[i,:] = np.concatenate((lstm2_out[i+5-5,:],lstm2_out[i+5-4,:],lstm2_out[i+5-3,:],lstm2_out[i+5-2,:],lstm2_out[i+5-1,:],lstm2_out[i+5,:],lstm2_out[i+5+1,:],lstm2_out[i+5+2,:],lstm2_out[i+5+3,:],lstm2_out[i+5+4,:],lstm2_out[i+5+5,:]),axis=0)
    
    speaker_out, dense3_out = model_spk(torch.from_numpy(lstm2_out_con.astype('float32')))

    enhanced_LP, _= model_enh(torch.from_numpy(n_sp.astype('float32')),dense3_out)

    # Transfer the output from specturm to waveform
    enhanced_LP = enhanced_LP.detach().numpy().squeeze()
    enhanced_wav = SPP_rec(np.exp(enhanced_LP.T) - 1,n_p)
    enhanced_wav = enhanced_wav/np.max(abs(enhanced_wav))
    sf.write(os.path.join("./new_dataset_lstm1out_mask_20epoch",noise_name,dB,wave_name), enhanced_wav, 16000)

    print("Created: %s, %s, %s"%(noise_name, dB, wave_name))

end_time = time.time()
print ('The testing for this file ran for %.2fm' % ((end_time - start_time) / 60.))
