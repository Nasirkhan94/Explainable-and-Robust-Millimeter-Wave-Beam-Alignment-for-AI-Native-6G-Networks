# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:06:53 2024

@author: NKHAN20
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from ComplexLayers_Torch import Beam_Classifier, DKNN
import torch.utils.data
from sklearn.model_selection import train_test_split
from beam_utils import ULA_DFT_codebook as DFT_codebook
from beam_utils import upa_codebook_generator_dft as UPA_DFT_codebook
from beam_utils import CustomDataset


np.random.seed(7)

n_wide_beams = n_wb= 32 
n_narrow_beams =n_nb= 128 

n_antenna = 32
batch_size = 512
noise_factor = -40 #dB

model_noise_power_dBm = -np.inf
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

    

learnable_codebook_topk_acc_vs_noise = []
dft_codebooks = []
learned_codebooks = []
AMCF_codebooks = []


h = h_real + 1j*h_imag

if dataset_name == 'O1_60':
    h=h.T
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor


train_idc, test_idc = train_test_split(np.arange(h_scaled.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)


dft_wb, all_beams=  UPA_DFT_codebook(1, 32, 1, over_sampling_x=1, over_sampling_y=1, over_sampling_z=1, ant_spacing=0.5) ## size # of antennas x codebooksize
dft_wb_codebook= dft_wb.T  
dft_upa, all_beams=  UPA_DFT_codebook(1, 32, 1, over_sampling_x=1, over_sampling_y=4, over_sampling_z=1, ant_spacing=0.5) ## size # of antennas x codebooksize
dft_nb_codebook= dft_upa.T
# dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna) # uniofrm DFT code book based labels


DNN_inputs= np.power(np.absolute(np.matmul(h_scaled, dft_wb_codebook.conj().T)),2) # (96850, 32)

label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1) # armax  label
soft_label = np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2) # actual  received powers over 128 narrowbeams (96850, 128)


x_train,y_train = DNN_inputs[train_idc,:],label[train_idc]
x_val,y_val = DNN_inputs[val_idc,:],label[val_idc]
x_test,y_test = DNN_inputs[test_idc,:],label[test_idc]


# Create dataset instances
train = CustomDataset(x_train, y_train, scale_factor=1, reshape_dims=(1, 8, 4), transform=None)
val = CustomDataset(x_val, y_val, scale_factor=1, reshape_dims=(1, 8, 4), transform=None)
test = CustomDataset(x_test, y_test, scale_factor=1, reshape_dims=(1, 8, 4), transform=None)
# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)


    
torch_x_val,torch_y_val = torch.from_numpy(x_val),torch.from_numpy(y_val)
torch_x_test,torch_y_test = torch.from_numpy(x_test),torch.from_numpy(y_test)
torch_x_test=torch_x_test.reshape(-1, 1, 8, 4)*1
torch_x_val=torch_x_val.reshape(-1, 1, 8, 4)*1


avg_opti_bf_gain_dB = 10*np.log10(np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2).max(axis=1).mean()) ## should be h because its the actual gain

target_avg_snr_dB = np.arange(-40,15,5)

avg_opti_bf_gain_dB = 10*np.log10(np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2).max(axis=1).mean()) ## should be h because its the actual gain

noise_power_dBm_array =  tx_power_dBm + avg_opti_bf_gain_dB - target_avg_snr_dB 
noise_power_array = 10**((noise_power_dBm_array)/10)



# Initialize lists to store top-k accuracies for different SNR levels
top1_acc_vs_snr = []  # Top-1 accuracy vs SNR
top3_acc_vs_snr = []  # Top-2 accuracy vs SNR
top5_acc_vs_snr = []  # Top-3 accuracy vs SNR

# Initialize lists to store top-k accuracies for different SNR levels for both noisy and non-noisy models
top1_acc_vs_snr_noisy = []  # Top-1 accuracy vs SNR (noisy model)
top3_acc_vs_snr_noisy = []  # Top-3 accuracy vs SNR (noisy model)
top5_acc_vs_snr_noisy = []  # Top-5 accuracy vs SNR (noisy model)

exhaustive_acc_vs_snr = []  # Exhaustive search accuracy vs SNR (with noise)
exhaustive_acc_vs_snr_wb=[]

for noise_power_dBm, noise_power in zip(noise_power_dBm_array, noise_power_array):
    
    learnable_codebook_model = DKNN()
    learnable_codebook_model.load_state_dict(torch.load('./Saved Models/{}_trainable_{}_sensing_codebook_{}_beam_classifier.pt'.format(dataset_name,n_wide_beams, n_narrow_beams)))
    
    # Noisy wide beamforming signal
    wb_bf_signal = np.matmul(h_scaled[test_idc], dft_wb_codebook.conj().T)
    best_wb = np.argmax(wb_bf_signal, axis=1)
    
    wb_bf_noise_real = np.random.normal(loc=0, scale=1, size=wb_bf_signal.shape) * np.sqrt(noise_power / 2)
    wb_bf_noise_imag = np.random.normal(loc=0, scale=1, size=wb_bf_signal.shape) * np.sqrt(noise_power / 2)
    wb_bf_signal_with_noise = wb_bf_signal + wb_bf_noise_real + 1j * wb_bf_noise_imag
    wb_bf_gain_with_noise = np.power(np.abs(wb_bf_signal_with_noise), 2)
    best_wb_noisy = np.argmax(wb_bf_gain_with_noise, axis=1)
    exhaustive_acc_vs_snr_wb.append(np.mean(best_wb_noisy == best_wb))  # Noisy exhaustive search accuracy

    torch_x_test = torch.from_numpy(wb_bf_gain_with_noise).reshape(-1, 1, 8, 4)
   
    y_test_predict_learnable_codebook = learnable_codebook_model(torch_x_test.float()).detach().numpy()
    topk_sorted_test_learned_codebook = (-y_test_predict_learnable_codebook).argsort()

    # Calculate top-k accuracies
    top1_acc_learnable_codebook = [(ue_bf_gain.argmax() in pred_sort[:1]) for ue_bf_gain, pred_sort in zip(soft_label[test_idc, :], topk_sorted_test_learned_codebook)]
    top3_acc_learnable_codebook = [(ue_bf_gain.argmax() in pred_sort[:3]) for ue_bf_gain, pred_sort in zip(soft_label[test_idc, :], topk_sorted_test_learned_codebook)]
    top5_acc_learnable_codebook = [(ue_bf_gain.argmax() in pred_sort[:5]) for ue_bf_gain, pred_sort in zip(soft_label[test_idc, :], topk_sorted_test_learned_codebook)]

    top1_acc_vs_snr.append(np.mean(top1_acc_learnable_codebook))
    top3_acc_vs_snr.append(np.mean(top3_acc_learnable_codebook))
    top5_acc_vs_snr.append(np.mean(top5_acc_learnable_codebook))

    #### Exhaustive search using OS-DFT###############
    nb_bf_signal = np.matmul(h_scaled[test_idc], dft_nb_codebook.conj().T)
    nb_bf_gain = np.power(np.abs(nb_bf_signal), 2)
    best_nb = np.argmax(nb_bf_gain, axis=1)
    # Noisy narrow beamforming signal
    nb_bf_noise_real = np.random.normal(loc=0, scale=1, size=nb_bf_signal.shape) * np.sqrt(noise_power / 2)
    nb_bf_noise_imag = np.random.normal(loc=0, scale=1, size=nb_bf_signal.shape) * np.sqrt(noise_power / 2)
    nb_bf_signal_with_noise = nb_bf_signal + nb_bf_noise_real + 1j * nb_bf_noise_imag
    nb_bf_gain_with_noise = np.power(np.abs(nb_bf_signal_with_noise), 2)

    best_nb_noisy = np.argmax(nb_bf_gain_with_noise, axis=1)
    exhaustive_acc_vs_snr.append(np.mean(best_nb_noisy == best_nb))  # Noisy exhaustive search accuracy
    
    #####################
    learnable_codebook_model_noisy = DKNN()
    learnable_model_savefname = './Saved Models/{}_trainable_{}_sensing_codebook_{}_new_beam_classifier_noisepower_{}_.pt'.format(dataset_name,n_wide_beams ,n_narrow_beams, noise_power_dBm)
    learnable_codebook_model_noisy.load_state_dict(torch.load(learnable_model_savefname))
    
    y_test_predict_learnable_codebook_noisy = learnable_codebook_model_noisy(torch_x_test.float()).detach().numpy()
    topk_sorted_test_learned_codebook_noisy = (-y_test_predict_learnable_codebook_noisy).argsort()

    # Calculate top-k accuracies
    
    # Calculate top-k accuracies (noisy model)
    top1_acc_learnable_codebook_noisy = [(ue_bf_gain.argmax() in pred_sort[:1]) for ue_bf_gain, pred_sort in zip(soft_label[test_idc, :], topk_sorted_test_learned_codebook_noisy)]
    top3_acc_learnable_codebook_noisy = [(ue_bf_gain.argmax() in pred_sort[:3]) for ue_bf_gain, pred_sort in zip(soft_label[test_idc, :], topk_sorted_test_learned_codebook_noisy)]
    top5_acc_learnable_codebook_noisy = [(ue_bf_gain.argmax() in pred_sort[:5]) for ue_bf_gain, pred_sort in zip(soft_label[test_idc, :], topk_sorted_test_learned_codebook_noisy)]

        # Store top-k accuracies for noisy model
    top1_acc_vs_snr_noisy.append(np.mean(top1_acc_learnable_codebook_noisy))
    top3_acc_vs_snr_noisy.append(np.mean(top3_acc_learnable_codebook_noisy))
    top5_acc_vs_snr_noisy.append(np.mean(top5_acc_learnable_codebook_noisy))



# Sort SNR values in increasing order and get corresponding indices
sorted_indices = np.argsort(target_avg_snr_dB)

# Sort the SNR values and corresponding accuracies using the sorted indices
sorted_snr = np.array(target_avg_snr_dB)[sorted_indices]
sorted_top5_acc_vs_snr_noisy = np.array(top5_acc_vs_snr_noisy)[sorted_indices] * 100.0
sorted_top3_acc_vs_snr_noisy = np.array(top3_acc_vs_snr_noisy)[sorted_indices] * 100.0
sorted_top1_acc_vs_snr_noisy = np.array(top1_acc_vs_snr_noisy)[sorted_indices] * 100.0
sorted_top5_acc_vs_snr = np.array(top5_acc_vs_snr)[sorted_indices] * 100.0
sorted_top3_acc_vs_snr = np.array(top3_acc_vs_snr)[sorted_indices] * 100.0
sorted_top1_acc_vs_snr = np.array(top1_acc_vs_snr)[sorted_indices] * 100.0
sorted_exhaustive_acc_vs_snr = np.array(exhaustive_acc_vs_snr)[sorted_indices] * 100.0
sorted_exhaustive_acc_vs_snr_wb = np.array(exhaustive_acc_vs_snr_wb)[sorted_indices] * 100.0


# Plotting top-k accuracy vs SNR in increasing order
plt.rcParams['figure.figsize'] = [6, 4.5]  # Width, Height in inches
plt.figure()

# Plot data for DkNN with noise
plt.plot(sorted_snr, sorted_top5_acc_vs_snr_noisy, marker='*', color='#EDB120', markersize=9, linewidth=1.5)
plt.plot(sorted_snr, sorted_top3_acc_vs_snr_noisy, marker='*', color='#EDB120', markersize=9, linewidth=1.5)
plt.plot(sorted_snr, sorted_top1_acc_vs_snr_noisy, marker='*', color='#EDB120', markersize=9, linewidth=1.5)

# Plot data for DkNN without noise
plt.plot(sorted_snr, sorted_top5_acc_vs_snr, marker='^', color='#0000FF', markersize=9, linewidth=1.5)
plt.plot(sorted_snr, sorted_top3_acc_vs_snr, marker='^', color='#0000FF', markersize=9, linewidth=1.5)
plt.plot(sorted_snr, sorted_top1_acc_vs_snr, marker='^', color='#0000FF', markersize=9, linewidth=1.5)

# Plot data for exhaustive search methods
plt.plot(sorted_snr, sorted_exhaustive_acc_vs_snr, marker='D', color='#00FF00', markersize=9, linewidth=1.5)
plt.plot(sorted_snr, sorted_exhaustive_acc_vs_snr_wb, marker='o', color='#FF2020', markersize=9, linewidth=1.5)

# Adding arrows with text {5,3,1} traversing curves for DkNN with noise
arrow_props = dict(facecolor='black', arrowstyle='->', lw=1.5)
plt.annotate('Top-{ 1, 3, 5}', xy=(sorted_snr[1]+1, sorted_top5_acc_vs_snr_noisy[1]+1.2), 
             xytext=(-30 -8, 50),
             arrowprops=arrow_props,fontsize=14)


# Set labels and grid
plt.xlabel('SNR (dB)', fontsize=15)
plt.ylabel('Accuracy (%)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)

# Adding a simplified legend in the center
handles = [
    plt.Line2D([0], [0], color='#EDB120', marker='*', markersize=9, label=r'DkNN$_{wn}$'),
    plt.Line2D([0], [0], color='#0000FF', marker='^', markersize=9, label=r'DkNN$_{wo}$'),
    plt.Line2D([0], [0], color='#00FF00', marker='D', markersize=9, label='O-DFT Codebook'),
    plt.Line2D([0], [0], color='#FF2020', marker='o', markersize=9, label='Classical DFT Codebook'),
]
leg = plt.legend(handles=handles, fontsize=14, loc='best', framealpha=0.6, frameon=True)
leg.get_frame().set_edgecolor('k')

# Optional: Save the plot
plots_folder = r'C:\Users\nkhan20\Desktop\Learning-Site-Specific-Probing-Beams-for-Fast-mmWave-Beam-Alignment-Public-main'
pdf_plot_path = os.path.join(plots_folder, 'topk_accuracy_vs_SNR_Boston_5G.eps')
plt.savefig(pdf_plot_path, format='eps', dpi=1000, bbox_inches='tight')

# Show the plot
plt.show()




#################

