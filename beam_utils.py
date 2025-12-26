

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import math


def evaluate_topk_accuracy(model, data_loader, k, use_cuda=True):
    model.eval()  # Set the model to evaluation mode

    if use_cuda and torch.cuda.is_available():
        model.cuda()

    topk_accuracies = {i: [] for i in range(1, k + 1)}

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for inputs, targets in data_loader:
            if use_cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = model(inputs)  # Forward pass to get the raw output (logits)
            top_beams, topk_indices = outputs.topk(k, dim=1, largest=True, sorted=True)  # Top-k predictions

            targets = targets.view(-1, 1)  # Convert targets to column vector

            for i in range(1, k + 1):
                topk_correct = (topk_indices[:, :i] == targets).sum().item()
                batch_size = targets.size(0)
                topk_accuracy = 100.0 * topk_correct / batch_size
                topk_accuracies[i].append(topk_accuracy)

    mean_topk_accuracies = {i: np.mean(topk_accuracies[i]) for i in topk_accuracies}
    return mean_topk_accuracies, top_beams



            
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, scale_factor=1, reshape_dims=(1, 8, 4), transform=None):
        self.transform = transform
        self.features = features * scale_factor  # Scaling features
        self.labels = labels
        self.reshape_dims = reshape_dims

        # Convert to tensors
        self.dataset = torch.tensor(self.features, dtype=torch.float32)
        
        # Reshape dataset if needed
        self.dataset = self.dataset.reshape(len(self.dataset), *self.reshape_dims)

        # Prepare labels
        self.labels = torch.tensor(self.labels, dtype=torch.long).reshape(-1)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

    def __len__(self):
        return len(self.dataset)



def upa_codebook_generator_dft(Mx, My, Mz, over_sampling_x=1, over_sampling_y=1, over_sampling_z=1, ant_spacing=0.5):
    """
    Generate a 3D Uniform Planar Array (UPA) DFT-based codebook for beamforming.

    Parameters:
    Mx: Number of antenna elements along the x-axis
    My: Number of antenna elements along the y-axis
    Mz: Number of antenna elements along the z-axis
    over_sampling_x: Sampling factor along the x-axis (default = 1)
    over_sampling_y: Sampling factor along the y-axis (default = 1)
    over_sampling_z: Sampling factor along the z-axis (default = 1)
    ant_spacing: Antenna spacing (default = 0.5, half-wavelength)

    Returns:
    F_CB: Complete 3D codebook (shape: [Mx * My * Mz, codebook_size_x * codebook_size_y * codebook_size_z])
    all_beams: All possible beam combinations (shape: [codebook_size_x * codebook_size_y * codebook_size_z, 3])
    """
    kd = 2 * np.pi * ant_spacing
    antx_index = np.arange(Mx)
    anty_index = np.arange(My)
    antz_index = np.arange(Mz)
    
    codebook_size_x = over_sampling_x * Mx
    codebook_size_y = over_sampling_y * My
    codebook_size_z = over_sampling_z * Mz

    # Generate DFT codebook for each dimension (x, y, z)
    theta_qx = np.linspace(0, 2 * np.pi, codebook_size_x, endpoint=False)
    F_CBx = np.sqrt(1 / Mx) * np.exp(-1j * np.outer(antx_index, theta_qx))

    theta_qy = np.linspace(0, 2 * np.pi, codebook_size_y, endpoint=False)
    F_CBy = np.sqrt(1 / My) * np.exp(-1j * np.outer(anty_index, theta_qy))

    theta_qz = np.linspace(0, 2 * np.pi, codebook_size_z, endpoint=False)
    F_CBz = np.sqrt(1 / Mz) * np.exp(-1j * np.outer(antz_index, theta_qz))

    # Kronecker products to create the full 3D codebook
    F_CBxy = np.kron(F_CBy, F_CBx)
    F_CB = np.kron(F_CBz, F_CBxy)  # Shape: [Mx * My * Mz, codebook_size_x * codebook_size_y * codebook_size_z]

    # Beam indices
    beams_x = np.arange(1, codebook_size_x + 1)
    beams_y = np.arange(1, codebook_size_y + 1)
    beams_z = np.arange(1, codebook_size_z + 1)

    Mxx_Ind = np.tile(beams_x, codebook_size_y * codebook_size_z)
    Myy_Ind = np.tile(np.repeat(beams_y, codebook_size_x), codebook_size_z)
    Mzz_Ind = np.repeat(beams_z, codebook_size_x * codebook_size_y)

    # All beam combinations
    all_beams = np.stack((Mxx_Ind, Myy_Ind, Mzz_Ind), axis=1)  # Shape: [codebook_size_x * codebook_size_y * codebook_size_z, 3]

    return F_CB, all_beams


def DFT_angles(n_beam):
    delta_theta = 1/n_beam
    if n_beam % 2 == 1:
        thetas = np.arange(0,1/2,delta_theta)
        # thetas = np.linspace(0,1/2,n_beam//2+1,endpoint=False)
        thetas = np.concatenate((-np.flip(thetas[1:]),thetas))
    else:
        thetas = np.arange(delta_theta/2,1/2,delta_theta) 
        thetas = np.concatenate((-np.flip(thetas),thetas))
    return thetas

def ULA_DFT_codebook(nseg,n_antenna,spacing=0.5):
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    thetas = DFT_angles(nseg)
    azimuths = np.arcsin(1/spacing*thetas)
    for i,theta in enumerate(azimuths):
        arr_response_vec = [-1j*2*np.pi*k*spacing*np.sin(theta) for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all

def DFT_beam(n_antenna,azimuths):
    codebook_all = np.zeros((len(azimuths),n_antenna),dtype=np.complex_)
    for i,phi in enumerate(azimuths):
        arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all


