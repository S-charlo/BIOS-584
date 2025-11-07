import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.io as sio # This will be used to load an MATLAB file
import os
from self_py_fun.HW8Fun import produce_trun_mean_cov, plot_trunc_mean, plot_trunc_cov


parent_dir = '/Users/sofia/Documents/GitHub/BIOS-584'
parent_data_dir = '/Users/sofia/Documents/GitHub/BIOS-584/data'
time_index = np.linspace(0, 800, 25) # This is a hypothetic time range up to 800 ms after each stimulus.

bp_low = 0.5
bp_upp = 6
electrode_num = 16
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

subject_name = 'K114'
session_name = '001_BCI_TRN'

eeg_trunc_obj = sio.loadmat('/Users/sofia/Documents/GitHub/BIOS-584/data/K114_001_BCI_TRN_Truncated_Data_0.5_6.mat')

eeg_trunc_signal = eeg_trunc_obj['Signal']
eeg_trunc_type = eeg_trunc_obj['Type'].flatten()
E_val = 16

output_dir = os.path.join(os.getcwd(), 'K114')

signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov = produce_trun_mean_cov(
    eeg_trunc_signal, eeg_trunc_type, E_val)

plot_trunc_mean(eeg_tar_mean=signal_tar_mean, eeg_ntar_mean=signal_ntar_mean, subject_name=subject_name, time_index=time_index, E_val=E_val, electrode_name_ls=electrode_name_ls, save_path=os.path.join(output_dir, 'Mean.png'))

plot_trunc_cov(eeg_cov=signal_tar_cov, cov_type="Target", subject_name=subject_name, time_index=time_index, E_val=E_val, electrode_name_ls=electrode_name_ls, save_dir=output_dir)

plot_trunc_cov(eeg_cov=signal_ntar_cov, cov_type="Non-Target", subject_name=subject_name, time_index=time_index, E_val=E_val, electrode_name_ls=electrode_name_ls, save_dir=output_dir)

plot_trunc_cov(eeg_cov=signal_all_cov, cov_type="All", subject_name=subject_name, time_index=time_index, E_val=E_val, electrode_name_ls=electrode_name_ls, save_dir=output_dir)


print("All figures saved to K114 folder.")
