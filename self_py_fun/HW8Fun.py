import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf


def produce_trun_mean_cov(eeg_trunc_signal, eeg_trunc_type, E_val):
    total_features = eeg_trunc_signal.shape[1]
    length_per_electrode = total_features // E_val

    tar_data = eeg_trunc_signal[eeg_trunc_type == 1, :]
    ntar_data = eeg_trunc_signal[eeg_trunc_type == -1, :]

    signal_tar_mean = np.zeros((E_val, length_per_electrode))
    signal_ntar_mean = np.zeros((E_val, length_per_electrode))
    signal_tar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_ntar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_all_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))

    for e in range(E_val):
        start_idx = e * length_per_electrode
        end_idx = start_idx + length_per_electrode

        tar_e = tar_data[:, start_idx:end_idx]
        ntar_e = ntar_data[:, start_idx:end_idx]
        all_e = eeg_trunc_signal[:, start_idx:end_idx]

        signal_tar_mean[e, :] = np.mean(tar_e, axis=0)
        signal_ntar_mean[e, :] = np.mean(ntar_e, axis=0)

        signal_tar_cov[e, :, :] = np.cov(tar_e, rowvar=False)
        signal_ntar_cov[e, :, :] = np.cov(ntar_e, rowvar=False)
        signal_all_cov[e, :, :] = np.cov(all_e, rowvar=False)

    return signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov


def plot_trunc_mean(
    eeg_tar_mean, eeg_ntar_mean, subject_name, time_index, E_val, electrode_name_ls,
    y_limit=np.array([-5, 8]), fig_size=(12, 12), save_path=None
):
    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    fig.suptitle(f"Subject: {subject_name}", fontsize=16, fontweight="bold")
    
    for i in range(E_val):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        
        ax.plot(time_index, eeg_tar_mean[i, :], color='red', label='Target')
        ax.plot(time_index, eeg_ntar_mean[i, :], color='blue', label='Non-Target')
        ax.set_title(electrode_name_ls[i])
        ax.set_xlabel("Time (ms)")  
        ax.set_ylabel("Amplitude (muV)")
        ax.set_ylim(y_limit)
    
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close(fig)


def plot_trunc_cov(
    eeg_cov, cov_type, time_index, subject_name, E_val, electrode_name_ls, fig_size=(12,12), save_dir=None
):
    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    fig.suptitle(f"{subject_name}{cov_type}: Covariance", fontsize=16, fontweight="bold")

    X, Y = np.meshgrid(time_index, time_index)
    
    for i in range(E_val):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        cov_matrix = eeg_cov[i, :, :]

        c = ax.contourf(X, Y, cov_matrix, cmap='viridis')
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    
        ax.set_title(electrode_name_ls[i])
        ax.set_xlabel("Time (ms)")  
        ax.set_ylabel("Time (ms)")
        ax.invert_yaxis()
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"Covariance_{cov_type}.png")
        plt.savefig(save_path)
    plt.close(fig)