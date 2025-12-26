# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:06:53 2024

@author: NKHAN20
"""
import numpy as np
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, Beam_Classifier_CNN, fit, eval_model,  DKNN
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils import upa_codebook_generator_dft as UPA_DFT_codebook
from beam_utils import CustomDataset

from beam_utils import evaluate_topk_accuracy
import os

# Create the directory if it doesn't exist
os.makedirs('./Saved Models', exist_ok=True)

n_wide_beams =  32 
n_narrow_beams = 128 
n_antenna = 32 # original is 64
nepoch = 100
batch_size = 512       
noise_factor = -13 #dB
noise_power_dBm = -np.inf ## means no noise in data
noiseless = False


dataset_name = 'Boston_5G' # 'Boston5G' ,'O1'
# dataset_name = 'O1_60' # 'Boston5G' ,'O1'
# Training and testing data:
if dataset_name == 'Boston_5G' : 
    h_real = np.load('./Dataset/Boston_5G/H_real_boston.npy')
    h_imag = np.load('./Dataset/Boston_5G/H_imag_boston.npy')
    tx_power_dBm = 30
elif dataset_name == 'O1_60':   
    h_real = np.load('./Dataset/O1_60/H_real_O1.npy')
    h_imag = np.load('./Dataset/O1_60/H_imag_O1.npy')
    tx_power_dBm = 30

else:
    raise NameError('Dataset Not Supported')   

if noiseless:
    noise_power_dBm = -np.inf    
noise_power = 10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)  


h = h_real + 1j*h_imag

if dataset_name == 'O1_60':
    h=h.T
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor


train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.7)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)  


dft_codebooks = []
learned_codebooks = []
AMCF_codebooks = []


dft_wb, all_beams=  UPA_DFT_codebook(1, 32, 1, over_sampling_x=1, over_sampling_y=1, over_sampling_z=1, ant_spacing=0.5) ## size # of antennas x codebooksize
dft_wb_codebook= dft_wb.T
    
dft_upa, all_beams=  UPA_DFT_codebook(1, 32, 1, over_sampling_x=1, over_sampling_y=4, over_sampling_z=1, ant_spacing=0.5) ## size # of antennas x codebooksize
dft_nb_codebook= dft_upa.T
# dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna) # uniofrm DFT code book based labels
label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1) # armax  label

# DNN_inputs= np.power(np.absolute(np.matmul(h_scaled, dft_wb_codebook.conj().T)),2) # (96850, 32)
# soft_label = np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2) # these are received powers over 128 narrowbeams (96850, 128)

# x_train,y_train = DNN_inputs[train_idc,:],label[train_idc]
# x_val,y_val = DNN_inputs[val_idc,:],label[val_idc]
# x_test,y_test = DNN_inputs[test_idc,:],label[test_idc]


# data= np.matmul(h_scaled, dft_nb_codebook.conj().T)

# # Create dataset instances
# # train = CustomDataset(x_train, y_train)
# # val = CustomDataset(x_val, y_val)
# # test = CustomDataset(x_test, y_test)

# # Create dataset instances
# train = CustomDataset(x_train, y_train, scale_factor=1, reshape_dims=(1, 8, 4), transform=None)
# val = CustomDataset(x_val, y_val, scale_factor=1, reshape_dims=(1, 8, 4), transform=None)
# test = CustomDataset(x_test, y_test, scale_factor=1, reshape_dims=(1, 8, 4), transform=None)


# # data loader
# train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
# val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
# test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)


#####

def add_noise(data, noise_power):
    """ Adds complex Gaussian noise to the data. """
    # Add real and imaginary noise
    noise_real = np.random.normal(loc=0, scale=1, size=data.shape) * np.sqrt(noise_power / 2)
    noise_imag = np.random.normal(loc=0, scale=1, size=data.shape) * np.sqrt(noise_power / 2)
    
    # Add real and imaginary parts of the noise to the signal
    noisy_data = data + noise_real + 1j * noise_imag  # Adding noise
    
    # Convert to magnitude (abs) of the noisy signal
    noisy_data_gain = np.power(np.abs(noisy_data), 2)
    
    return noisy_data_gain

#####

# noise_power_dBm_array=np.arange(-70,10,10)-30

# noise_power_array = 10**((noise_power_dBm_array)/10)

target_avg_snr_dB = np.arange(-50,15,5)

avg_opti_bf_gain_dB = 10*np.log10(np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2).max(axis=1).mean()) ## should be h because its the actual gain

noise_power_dBm_array =  tx_power_dBm + avg_opti_bf_gain_dB - target_avg_snr_dB 
# noise_power_dBm_array = np.insert(noise_power_dBm_array,0,-np.inf)
noise_power_array = 10**((noise_power_dBm_array)/10)



for noise_power_dBm, noise_power in zip(noise_power_dBm_array, noise_power_array):
    
    print("TRAINING FOR noise_power_dBm: ", noise_power_dBm)
    
    data= np.matmul(h_scaled, dft_wb_codebook.conj().T)
    
    x_train,y_train = data[train_idc,:],label[train_idc]
    x_val,y_val = data[val_idc,:],label[val_idc]
    x_test,y_test =data[test_idc,:],label[test_idc]

    # Convert original data to noisy data
    x_train_noisy = add_noise(x_train, noise_power)
    x_val_noisy = add_noise(x_val, noise_power)
    x_test_noisy = add_noise(x_test, noise_power)
    
    # Convert noisy data to torch tensors and reshape
    x_train_noisy = torch.from_numpy(x_train_noisy).float().reshape(-1, 1, 8, 4)  # Assuming shape (1, 8, 4)
    x_val_noisy = torch.from_numpy(x_val_noisy).float().reshape(-1, 1, 8, 4)
    x_test_noisy = torch.from_numpy(x_test_noisy).float().reshape(-1, 1, 8, 4)
    
    # Datasets with noisy data
    train_noisy = CustomDataset(x_train_noisy, y_train, scale_factor=1, reshape_dims=(1, 8, 4), transform=None)
    val_noisy = CustomDataset(x_val_noisy, y_val, scale_factor=1, reshape_dims=(1, 8, 4), transform=None)
    test_noisy = CustomDataset(x_test_noisy, y_test, scale_factor=1, reshape_dims=(1, 8, 4), transform=None)
    
    # Noisy data loader
    train_loader_noisy = torch.utils.data.DataLoader(train_noisy, batch_size=batch_size, shuffle=False)
    val_loader_noisy = torch.utils.data.DataLoader(val_noisy, batch_size=batch_size, shuffle=False)
    test_loader_noisy = torch.utils.data.DataLoader(test_noisy, batch_size=batch_size, shuffle=False)
    
    
    learnable_codebook_model= DKNN()
    learnable_codebook_opt = optim.Adam(learnable_codebook_model.parameters(),lr=0.001, betas=(0.9,0.999), amsgrad=False)
    train_loss_hist, val_loss_hist = fit(learnable_codebook_model, train_loader_noisy, val_loader_noisy, learnable_codebook_opt, nn.CrossEntropyLoss(), nepoch)  
    learnable_model_savefname = './Saved Models/{}_trainable_{}_sensing_codebook_{}_new_beam_classifier_noisepower_{}_.pt'.format(dataset_name,n_wide_beams ,n_narrow_beams, noise_power_dBm)
    torch.save(learnable_codebook_model.state_dict(),learnable_model_savefname)
    plt.figure()
    plt.plot(train_loss_hist,label='training loss')
    plt.plot(val_loss_hist,label='validation loss')
    plt.legend()
    # plt.title('Trainable codebook loss hist: {} wb {} nb'.format(n_wb,n_nb))
    plt.show()
    # learned_codebooks.append(learnable_codebook_model.get_codebook()) 
    
    top_k, top_k_logits = evaluate_topk_accuracy(learnable_codebook_model, test_loader_noisy, 3, use_cuda=True)
    print("topk_accuracy is",top_k)
    print("top_k_logits  are",top_k_logits )

  