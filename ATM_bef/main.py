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
from util import creatdir, get_filepaths, SPP_make, SPP_rec, read_wav, data_generator, valid_generator
from new_dataset_lstm1out_mask_model import Enhance, Speaker

os.environ["CUDA_VISIBLE_DEVICES"]="2" #Your GPU number, default = 0

random.seed(999)

batch_size= 1
Epoch = 20

################ Read data #######################################
clean_path = "..."    # You need to write your own clean data path 
speaker_label = pd.read_csv('TMHINT_new_train_label.csv')   # You need to write your own clean speaker label

noisy_list = np.load('noisy_list_new.npy')  # read the noisy data from noisy_list_new.npy (have been shuffle already)
noisy_list = noisy_list.tolist()

idx = int(len(noisy_list)*0.95)
Train_list = noisy_list[0:idx]
Num_traindata = len(Train_list)
Valid_list = noisy_list[idx:]

steps_per_epoch = (Num_traindata)//batch_size
Num_testdata=len(Valid_list)


######################### Training Stage ########################           
start_time = time.time()

print('model building...')

######## Initialize models weights and set the loss functions #######
def intialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight.data)

enh_model = Enhance().cuda()
enh_model.apply(intialize_weights)

speaker_model = Speaker().cuda()
speaker_model.apply(intialize_weights)

criterion_enh = nn.MSELoss()
criterion_speaker = nn.CrossEntropyLoss()

optimizer_enh = torch.optim.Adam(enh_model.parameters(), lr=0.0005)
optimizer_speaker = torch.optim.Adam(speaker_model.parameters(), lr=0.0005)


######## Start training ############
print('training...')
loss_enh = []
loss_speaker = []
loss_enh_val = []
loss_speaker_val = []
loss_enh_val_avg = []
loss_speaker_val_avg = []
acc = []
Acc_epoch = []
count = 50

for epoch in range(Epoch):
    random.shuffle(Train_list)
    loss_enh_avg = []
    loss_speaker_avg = []
    
    for iter in tqdm(range(len(Train_list))):
        if iter % 50 == 0 and iter > 1:
            count = iter

        n_sp , c_sp , frames_label = data_generator(Train_list, clean_path, iter, speaker_label)

        enh_model.train()
        speaker_model.train()
        optimizer_enh.zero_grad()
        optimizer_speaker.zero_grad()

        # Doing the enhance model first time
        enh_out, lstm2_out = enh_model(torch.from_numpy(n_sp.astype('float32')).cuda() , None)

        # concat lstm output for speaker model input
        lstm2_out = lstm2_out.cpu().detach().numpy()
        lstm2_out = lstm2_out.squeeze()
        lstm2_out = np.pad(lstm2_out,((5,5),(0,0)),'reflect')
        lstm2_out_con = np.zeros([624,3300]).astype('float32')
        for i in range(624):
            lstm2_out_con[i,:] = np.concatenate((lstm2_out[i+5-5,:],lstm2_out[i+5-4,:],lstm2_out[i+5-3,:],lstm2_out[i+5-2,:],lstm2_out[i+5-1,:],lstm2_out[i+5,:],lstm2_out[i+5+1,:],lstm2_out[i+5+2,:],lstm2_out[i+5+3,:],lstm2_out[i+5+4,:],lstm2_out[i+5+5,:]),axis=0)
        
        # Receiving speaker embedding and do the enhance model second time
        speaker_out, dense3_out = speaker_model(torch.from_numpy(lstm2_out_con.astype('float32')).cuda())

        enh_out, lstm2_out = enh_model(torch.from_numpy(n_sp.astype('float32')).cuda() , dense3_out.detach().cuda())
        
        # Computing loss and doing the backpropagation
        loss_e = criterion_enh(torch.squeeze(enh_out),torch.squeeze(torch.from_numpy(c_sp.astype('float32'))).cuda())
        loss_e.backward()
        optimizer_enh.step()

        loss_enh.append(loss_e.cpu().item())
        loss_enh_avg.append(loss_e.cpu().item())

        loss_s = criterion_speaker(speaker_out,torch.from_numpy(frames_label).type(torch.LongTensor).cuda())

        loss_s.backward()
        optimizer_speaker.step()
        
        loss_speaker.append(loss_s.cpu().item())
        loss_speaker_avg.append(loss_s.cpu().item())

    
    loss_enh_avg = sum(loss_enh_avg)/len(loss_enh_avg)
    loss_speaker_avg = sum(loss_speaker_avg)/len(loss_speaker_avg)
    print("Epoch(%d/%d): enh_loss %f spk_loss %f" % (epoch+1, Epoch, loss_enh_avg,loss_speaker_avg))

            
######### Validation ############################
    with torch.no_grad():
        enh_model.eval()
        speaker_model.eval()
        for iter in range(len(Valid_list)):
            n_sp , c_sp , frames_label = valid_generator(Valid_list, clean_path, iter, speaker_label)

            # Doing the enhance model first time
            enh_out, lstm2_out = enh_model(torch.from_numpy(n_sp.astype('float32')).cuda() , None)

            # concat lstm output for speaker model input
            lstm2_out = lstm2_out.cpu().detach().numpy()
            lstm2_out = lstm2_out.squeeze()
            lstm2_out = np.pad(lstm2_out,((5,5),(0,0)),'reflect')
            lstm2_out_con = np.zeros([624,3300]).astype('float32')
            for i in range(624):
                lstm2_out_con[i,:] = np.concatenate((lstm2_out[i+5-5,:],lstm2_out[i+5-4,:],lstm2_out[i+5-3,:],lstm2_out[i+5-2,:],lstm2_out[i+5-1,:],lstm2_out[i+5,:],lstm2_out[i+5+1,:],lstm2_out[i+5+2,:],lstm2_out[i+5+3,:],lstm2_out[i+5+4,:],lstm2_out[i+5+5,:]),axis=0)

            # Receiving speaker embedding and do the enhance model second time
            speaker_out, dense3_out = speaker_model(torch.from_numpy(lstm2_out_con.astype('float32')).cuda())
            enh_out, lstm2_out = enh_model(torch.from_numpy(n_sp.astype('float32')).cuda() , dense3_out.detach().cuda())

            # Computing loss
            loss_s = criterion_speaker(speaker_out,torch.from_numpy(frames_label).type(torch.LongTensor).cuda())
            loss_speaker_val.append(loss_s.cpu().item())

            loss_e = criterion_enh(torch.squeeze(enh_out),torch.squeeze(torch.from_numpy(c_sp)).cuda())
            loss_enh_val.append(loss_e.cpu().item())

            # Computing soeaker accuracy
            running_acc_initial = 0.0
            prediction = torch.argmax(speaker_out, dim=1)
            running_acc_initial += torch.sum(prediction == torch.from_numpy(frames_label).cuda())
            acc.append((running_acc_initial/len(frames_label)))

        loss_enh_val_avg.append(sum(loss_enh_val)/len(loss_enh_val))
        loss_speaker_val_avg.append(sum(loss_speaker_val)/len(loss_speaker_val))
        Acc_epoch.append((sum(acc)/len(acc)).item())
        print("Acc : %f enh_loss : %f" %((sum(acc)/len(acc)),(sum(loss_enh_val)/len(loss_enh_val))))
    
    # Finished running an epoch and save the model
    Path = 'new_dataset_lstm1out_mask' + str(epoch) + '.pth'
    torch.save(enh_model.state_dict(),'ENHANCE_'+Path)
    torch.save(speaker_model.state_dict(),'SPEAKER_'+Path)


for i in range(len(loss_speaker_val_avg)):
    print("Epoch:%d enh_val_avg:%f speaker_val_avg:%f \n"%(i,loss_enh_val_avg[i],loss_speaker_val_avg[i]))

np.save('new_dataset_lstm1out_mask_ACC.npy', np.array(Acc_epoch))
np.save('new_dataset_lstm1out_mask_enh_loss.npy', np.array(loss_enh_val_avg))
np.save('new_dataset_lstm1out_mask_speaker_loss.npy', np.array(loss_speaker_val_avg))
     
