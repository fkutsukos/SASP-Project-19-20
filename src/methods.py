import numpy as np
from scipy.signal import find_peaks
import sympy as sp
# noinspection PyDeprecation
import pyroomacoustics as pra
import json
import os


# noinspection PyDeprecation
def build_room(dimensions, source, mic_array, rt60, fs):
    """
    This method wraps inside all the necessary steps
    to build the simulated room
    :param dimensions: length, width and height of the room
    :param source: contains the x, y, z location of the sound source
    :param mic_array: contains the x, y, z vectors of all the microphones
    :param float rt60: represents the reverberation time of the room
    :param int fs: represents the sampling frequency used for the generation of signals
    :return: pyroomacoustics object representing the room and the fractional delay to compensate
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

    # Getting the fractional delay introduced by the simulation
    global_delay = pra.constants.get("frac_delay_length") // 2

    return room, global_delay


def peak_picking(mic_rirs, number_of_echoes, prominence=0.05, distance=5):
    """
    This method receives a list of microphone room impulse responses and extracts the location of the main peaks.
    Impulse responses contain peaks that do not correspond to any wall.
    These spurious peaks can be introduced by noise, non-linearities, and other imperfections in the measurement system.
    :param mic_rirs: list of microphone room impulse responses
    :param number_of_echoes: maximum number of peaks to identify, a good strategy is to select a number of peaks greater
    than the number of walls and then to prune the selection.
    :param prominence: required prominence of peaks.
    :param distance: required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    :return: list of arrays that contain the peak indexes
    """
    peak_indexes = []
    for mic in range(len(mic_rirs)):
        peaks, _ = find_peaks(mic_rirs[mic][0], prominence=prominence, distance=distance)
        peak_indexes.append(peaks[0:number_of_echoes])
    return peak_indexes


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
        for j in range(i + 1, n):
            D[i, j] = np.linalg.norm(mic_locations[:, i] - mic_locations[:, j]) ** 2
            D[j, i] = D[i, j]
    return D


def echo_labeling(edm, mic_peaks, k, fs, global_delay, c=343.0):
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
    :param global_delay: fractional delay to be compensated.
    :param c: speed of sound (set to the default value of 343 m/s).
    :return: a list containing all the echo combinations that satisfy
    the EDM property
    """
    # Converting the RIR peak location indexes into space distances
    mic_peaks -= global_delay
    mic_peaks *= c / fs

    # Building a grid of each possible echo combination
    echoes_comb = np.array(np.meshgrid(*mic_peaks)).T.reshape(-1, len(mic_peaks))

    # Getting the input EDM dimensions and setting Daug dimensions
    dim = edm.shape[0]
    dim += 1

    # Instantiating the Daug matrix
    d_aug = np.zeros(shape=(dim, dim))

    # Start building Daug inserting the initial EDM
    d_aug[0:dim - 1, 0:dim - 1] = edm
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


def trilateration(mic_locations, distances):
    """
    This method uses a Trilateration algorithm to infer the position
    of the virtual source corresponding to a set of measured echoes
    using the location of the microphones and the distance between each
    microphone and the virtual source (estimated from the RIRs).
    :param mic_locations: contains the x, y, z vectors of all the microphones
    :param distances: contains the distances between each microphone
    and the virtual source to be estimated
    :return: the x, y, z coordinates of the virtual source
    """
    # Trilateration using a Linearized System of Equations:
    # Conventionally, we use the first mic as a reference point
    reference_mic = mic_locations[:, 0]

    # Instantiating the A matrix, which has a number of rows = n_mics - 1
    # and a number of columns = n_dimensions
    A = np.zeros(shape=(mic_locations.shape[1] - 1, len(mic_locations)))

    # Building the A matrix
    for i in range(len(reference_mic)):
        for j in range(mic_locations.shape[1] - 1):
            A[j, i] = mic_locations[i, j + 1] - reference_mic[i]

    # Instantiating the b vector, which has dimension = n_mics - 1
    b = np.zeros(mic_locations.shape[1] - 1)

    # Building the b vector
    for i in range(mic_locations.shape[1] - 1):
        b[i] = 0.5 * (distances[0] ** 2 - distances[i + 1] ** 2 +
                      np.linalg.norm(mic_locations[:, i + 1] - reference_mic) ** 2)

    # Solving the linear system through the Linear Least Squares (Moore-Penrose Pseudo-inverse) approach
    virtual_source_loc = np.linalg.pinv(A).dot(b) + reference_mic

    return virtual_source_loc


def reconstruct_room(candidate_virtual_sources, loudspeaker, dist_thresh):
    """
    This method uses the first-order virtual-sources to reconstruct the room:
    it processes the candidate virtual sources in the order of increasing distance
    from the loudspeaker to find the first-order virtual sources and add their planes
    to the list of half-spaces whose intersection determines the final room.
    :param candidate_virtual_sources: list of the coordinates of all the individuated
    virtual sources (it could contain even higher-order virtual sources)
    :param loudspeaker: x, y, z coordinates of the speaker location in the room
    :param dist_thresh: distance threshold (epsilon)
    :return: list of planes corresponding to the first-order virtual sources
    """

    def combine(s1, s2):
        """
        This method combines the virtual sources s1 and s2 to generate a higher-order
        virtual source; it is used as a criterion to discard higher-order virtual sources.
        :param s1: first virtual source coordinates
        :param s2: second virtual source coordinates
        :return: the coordinates of the higher order virtual source generated through
        the combination of s1 and s2
        """
        # p2 is a point on the hypothetical wall defined by s2, that is, a point on
        # the median plane between the loudspeaker and s2
        p2 = (loudspeaker + s2) / 2

        # n2 is the outward pointing unit normal
        n2 = (s2 - loudspeaker) / np.linalg.norm(s2 - loudspeaker)

        return s1 + 2 * np.dot((p2 - s1), n2) * n2

    # Instantiating the array to contain the distance of each virtual source from the loudspeaker
    distances_from_speaker = []

    # Computing the distances
    for source in candidate_virtual_sources:
        distances_from_speaker.append(np.linalg.norm(source - loudspeaker))

    # Re-ordering the list of virtual sources according to their distance from the loudspeaker
    distances_from_speaker = np.array(distances_from_speaker)
    sorted_virtual_sources = candidate_virtual_sources[distances_from_speaker.argsort()]

    # Initialize the list of planes that constitutes the room
    room = []

    # Initialize the boolean mask to identify the first-order virtual sources
    deleted = np.array([False] * len(sorted_virtual_sources), dtype=bool)

    for i in range(len(sorted_virtual_sources)):
        for idx1 in range(i):
            for idx2 in range(i):
                # The following two conditions verify if the current virtual source is a combination of lower order
                # virtual sources: if so, it is deleted from the available candidates
                if idx1 != idx2 and not deleted[idx1] and not deleted[idx2]:
                    if np.linalg.norm(
                            combine(
                                sorted_virtual_sources[idx1], sorted_virtual_sources[idx2]
                            ) - sorted_virtual_sources[i]
                    ) < dist_thresh:
                        deleted[i] = True
                    # If the virtual source is not a combination of lower order virtual sources, the corresponding plane
                    # is built and it is added to the room's walls list
                    else:
                        # pi is a point on the hypothetical wall defined by si, that is, a point on
                        # the median plane between the loudspeaker and si
                        pi = (loudspeaker + sorted_virtual_sources[i]) / 2

                        # n2 is the outward pointing unit normal
                        ni = (sorted_virtual_sources[i] - loudspeaker) / np.linalg.norm(
                            sorted_virtual_sources[i] - loudspeaker)

                        plane = sp.Plane(p1=pi, normal_vector=ni)
                        for wall in room:
                            if plane.intersection(wall) > 0:
                                room.append(plane)
                                break

                # TODO finish the following method implementation:
                # else if Plane(si) intersects the current room
                # add Plane(si) to the set of planes
                # else
                # deleted[i] = true
    return room


def input_data(file_dir="../input", file_name="room.json"):
    """
    This method receives the input data from a JSON formatted file
    :param: file_dir: the directory from where to read the data
    :param: file_name: the name of the file to be read
    :return: the dictionary of input data
    """
    data = {}
    path_to_file = os.path.join(file_dir, file_name)
    if os.path.exists(path_to_file):
        with open(path_to_file) as f:
            data = json.load(f)
    else:
        print('File not found in directory {}'.format(file_dir))
    return data
