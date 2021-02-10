import numpy as np


def find_echoes(mic_rirs, n=7, c=343, fs=96000, upsampling_rate=1, global_delay=40):
    echoes = np.zeros([len(mic_rirs), n])
    for j in range(len(mic_rirs)):
        direct_sound_index, side_lobe_rng, all_peak_indices = direct_sound(mic_rirs[j][0])
        N = direct_sound_index + side_lobe_rng
        peak_indices = all_peak_indices[np.where(all_peak_indices > N)[0]]
        i = np.argsort(mic_rirs[j][0][peak_indices])
        sorted_peak_indices = peak_indices[i][::-1]
        echoes[j, :] = np.r_[direct_sound_index, sorted_peak_indices[:(n - 1)]]

    # echoes -= global_delay
    # echoes = echoes * c / (upsampling_rate * fs)
    return echoes


def direct_sound(x):
    all_peak_indices = local_max(x, threshold=1e-5)
    i = np.argsort(x[all_peak_indices])
    direct_sound_index = all_peak_indices[i][-1]
    peak_indices = all_peak_indices[np.where(all_peak_indices < direct_sound_index)[0]]
    values = x[peak_indices]
    # direct sound side lobe range
    rng = direct_sound_index - max(peak_indices[np.where(values < (x[direct_sound_index] * 0.02))[0]])
    return direct_sound_index, rng, all_peak_indices


def local_max(x, threshold=1e-5):
    """
        Get all local maxima of x by selecting all points which are
        higher than its left and right neighbour
    """
    maxima = np.r_[True, x[1:] > x[:-1]] & np.r_[x[:-1] > x[1:], True]
    # select all local maxima above the threshold
    maxima_f = maxima & np.r_[x > threshold, True][:-1]
    peak_indices = np.where(maxima_f == True)[0]
    return np.array(peak_indices)
