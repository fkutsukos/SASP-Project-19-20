import numpy as np


def find_echoes(mic_rirs, n=7):
    """
    This method receives a list of microphone room impulse responses and extracts the location of the main peaks.
    Impulse responses contain peaks that do not correspond to any wall.
    These spurious peaks can be introduced by noise, non-linearities, and other imperfections in the measurement system.
    :param mic_rirs: list of microphone room impulse responses
    :param n: maximum number of peaks to identify, a good strategy is to select a number of peaks greater
    than the number of walls and then to prune the selection.
    :return: list of arrays that contain the peak indexes
    """
    echoes = np.zeros([len(mic_rirs), n])
    for j in range(len(mic_rirs)):
        direct_sound_index, side_lobe_rng, all_peak_indices = direct_sound(mic_rirs[j][0])
        N = direct_sound_index + side_lobe_rng
        peak_indices = all_peak_indices[np.where(all_peak_indices > N)[0]]
        i = np.argsort(mic_rirs[j][0][peak_indices])
        sorted_peak_indices = peak_indices[i][::-1]
        echoes[j, :] = np.r_[direct_sound_index, sorted_peak_indices[:(n - 1)]]

    return echoes


def direct_sound(x):
    """
    This method estimates the index of the direct sound coming from the
    loudspeaker to a given microphone.
    :param x: RIR of a given microphone
    :return: The index of the direct sound, the range of indexes that constitute the peak's side lobes, the indexes of
    all the other peaks
    """
    all_peak_indices = local_max(x, threshold=1e-5)
    i = np.argsort(x[all_peak_indices])
    direct_sound_index = all_peak_indices[i][-1]
    peak_indices = all_peak_indices[np.where(all_peak_indices < direct_sound_index)[0]]
    values = x[peak_indices]
    # Direct sound side lobe range
    rng = direct_sound_index - max(peak_indices[np.where(values < (x[direct_sound_index] * 0.02))[0]])
    return direct_sound_index, rng, all_peak_indices


def local_max(x, threshold=1e-5):
    """
    Get all local maxima of x by selecting all points which are
    higher than its left and right neighbour
    :param x: given RIR
    :param threshold: fixed lower bound for maxima retrieval
    :return: Array containing all the detected peaks
    """
    maxima = np.r_[True, x[1:] > x[:-1]] & np.r_[x[:-1] > x[1:], True]
    # Select all local maxima above the threshold
    maxima_f = maxima & np.r_[x > threshold, True][:-1]
    peak_indices = np.where(maxima_f)[0]
    return np.array(peak_indices)
