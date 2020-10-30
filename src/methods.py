import numpy as np
import pyroomacoustics as pra


def build_room(dimensions, source, mic_array, rt60, fs):
    """
    This method wraps inside all the necessary steps
    to build the simulated room
    :param dimensions: length, width and height of the room
    :param source: contains the x, y, z location of the sound source
    :param mic_array: contains the x, y, z vectors of all the microphones
    :param float rt60: represents the reverberation time of the room
    :param int fs: represents the sampling frequency used for the generation of signals
    :return: pyroomacoustics object representing the room
    """
    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60, dimensions)

    # Building a 'Shoebox' room with the provided dimensions
    room = pra.ShoeBox(p=dimensions, fs=fs, absorption=e_absorption, max_order=max_order)

    # Place the Microphone Array and the Sound Source inside the room
    mics = pra.MicrophoneArray(mic_array, fs)
    room.add_microphone_array(mics)
    room.add_source(source)

    # Computing the Room Impulse Response at each microphone, for each source
    room.image_source_model()
    room.compute_rir()

    return room


def peakPicking(mic_rir, numberOfEchoes):
    return peaksArray


def build_edm(mic_locations):
    """
    This method builds the Euclidean Distance Matrix using
    the relative positions between the microphones
    :param mic_locations: contains the x, y, z vectors of all the microphones
    :return: EDM matrix based on the distance between microphones
    """
    # Determine the dimensions of the input matrix
    m, n = mic_locations.shape

    # Initializing squared EDM
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = np.linalg.norm(mic_locations[:, i] - mic_locations[:, j])**2
            D[j, i] = D[i, j]
    return D


def echo_labeling(edm, mic_peaks, k, fs, c=343.0):
    """
    This method evaluates if the matrix Daug, generated adding a row and
    a column made of the possible echo combinations squared to the
    EDM matrix built over the microphone distances is still an
    Euclidean Distance Matrix;
    :param edm: Euclidean Distance Matrix built using the distances
    between the different microphones.
    :param mic_peaks: list of arrays containing the indexes of the
    peaks found in the RIR captured by each microphone.
    :param k: dimensionality of the space.
    :param fs: sampling frequency used in the RIRs measure.
    :param c: speed of sound (set to the default value of 343 m/s).
    :return: a list containing all the echo combinations that satisfy
    the EDM property
    """
    # Converting the RIR peak location indexes into space distances
    mic_peaks *= c/fs

    # Building a grid of each possible echo combination
    echoes_comb = np.array(np.meshgrid(*mic_peaks)).T.reshape(-1, len(mic_peaks))

    # Getting the input EDM dimensions and setting Daug dimensions
    dim = edm.shape[0]
    dim += 1

    # Instantiating the Daug matrix
    d_aug = np.zeros(shape=(dim, dim))

    # Start building Daug inserting the initial EDM
    d_aug[0:dim-1, 0:dim-1] = edm
    # Instantiate the list of matching echoes
    echoes_match = []

    # Echo labeling process
    for echo in echoes_comb:
        # Adding the echo combination squared
        d_aug[0:-1, -1] = echo ** 2
        d_aug[-1, 0:-1] = echo ** 2
        print(d_aug)
        # In a k dimensional space, the rank cannot be greater than k + 2
        if np.linalg.matrix_rank(d_aug) <= k + 2:
            print(np.linalg.matrix_rank(d_aug))
            echoes_match.append(echo)
    print(echoes_match)
    return echoes_match


def trilateration(sortedEchoestoImageSources):
    # using Linear Least Squares
    # using System of Linear Equations
    return
