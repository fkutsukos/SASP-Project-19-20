import numpy as np
from scipy.signal import find_peaks
import sympy as sp
from scipy.optimize import minimize
from scipy import interpolate
import pyroomacoustics as pra
import json
import os
import matplotlib.pyplot as plt

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
    room = pra.ShoeBox(p=dimensions, fs=fs, absorption=e_absorption, max_order=2)

    # Place the Microphone Array and the Sound Source inside the room
    mics = pra.MicrophoneArray(mic_array, fs)
    room.add_microphone_array(mics)
    room.add_source(source)

    # Computing the Room Impulse Response at each microphone, for each source
    room.image_source_model()
    room.compute_rir()

    # Getting the fractional delay introduced by the simulation
    global_delay = pra.constants.get("frac_delay_length") // 2

    # room.plot()

    return room, global_delay


def spline_interpolation(mic_rir, N):
    l = len(mic_rir)
    x = np.linspace(0, l, l)
    spline = interpolate.InterpolatedUnivariateSpline(x, mic_rir)
    x_new = np.linspace(0, l, N * l)
    y_new = spline(x_new)
    return y_new


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
        # peaks, _ = find_peaks(mic_rirs[mic], prominence=prominence, distance=distance)
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

        # print(d_aug)
        # In a k dimensional space, the rank cannot be greater than k + 2
        # if np.linalg.matrix_rank(d_aug) <= k + 2:
        # print(np.linalg.matrix_rank(d_aug))
        # embedding = MDS(n_components=k, dissimilarity="precomputed")
        # embedding.fit(d_aug)

        def stress(x, daug):
            m = len(daug)
            X = x.reshape((-1, m))
            stress_score = 0
            for j in range(m):
                for i in range(m):
                    stress_score = stress_score + (np.linalg.norm(X[:, j] - X[:, i]) ** 2 - daug[i, j]) ** 2

            return stress_score

        res = minimize(stress, x0=np.random.rand(1, len(d_aug) * 3), args=(d_aug,), method='SLSQP',
                       options={'disp': False})

        # print(embedding.embedding_)
        # print(embedding.stress_)
        score = res.fun
        echoes_match.append((score, echo))
    # print(echoes_match)
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
        n2 = (loudspeaker - s2) / np.linalg.norm(loudspeaker - s2)

        return s1 + 2 * np.dot((p2 - s1), n2) * n2

    def perpendicular(a):
        """
        This method...
        :param a:
        :return:
        """
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    # Instantiating the array to contain the distance of each virtual source from the loudspeaker
    distances_from_speaker = []

    # Computing the distances
    for source in candidate_virtual_sources:
        distances_from_speaker.append(np.linalg.norm(source - loudspeaker))

    # Re-ordering the list of virtual sources according to their distance from the loudspeaker
    candidate_virtual_sources = np.array(candidate_virtual_sources)
    sorted_virtual_sources = candidate_virtual_sources[np.array(distances_from_speaker).argsort()][1:]

    # wall_points = calculate_wall_points(sorted_virtual_sources, loudspeaker)

    # vertices = calculate_vertices2D(wall_points)

    # Initialize the list of planes that constitutes the room
    room = []
    vertices = []
    # Initialize the boolean mask to identify the first-order virtual sources
    deleted = np.array([False] * len(sorted_virtual_sources), dtype=bool)

    for i in range(len(sorted_virtual_sources)):
        for j in range(i):
            for k in range(i):
                # The following two conditions verify if the current virtual source is a combination of lower order
                # virtual sources: if so, it is deleted from the available candidates
                if j != k and k < i:
                    if np.linalg.norm(
                            combine(
                                sorted_virtual_sources[j], sorted_virtual_sources[k]
                            ) - sorted_virtual_sources[i]
                    ) < dist_thresh:
                        deleted[i] = True

                    # If the virtual source is not a combination of lower order virtual sources, the corresponding plane
                    # is built and it is added to the room's walls list

    for i in range(len(sorted_virtual_sources)):
        if deleted[i] is True:
            continue
        else:
            # pi is a point on the hypothetical wall defined by si, that is, a point on
            # the median plane between the loudspeaker and si
            pi = (loudspeaker + sorted_virtual_sources[i]) / 2
            # ni is the outward pointing unit normal
            ni = (loudspeaker - sorted_virtual_sources[i]) / np.linalg.norm(
                loudspeaker - sorted_virtual_sources[i])

            plane = {}
            # plane = sp.Plane(p1=pi, normal_vector=ni)
            if len(pi) == 2:
                ni_perp = perpendicular(ni)
                pi2 = pi + ni_perp
                plane = sp.Line(pi, pi2)

            elif len(pi) == 3:
                plane = sp.Plane(sp.Point3D(pi), normal_vector=ni)

            # If the room is empty, we add the first plane to the list of halfspaces whose intersection
            # determines the final room
            if len(room) == 0:
                room.append(plane)
            else:
                for wall in room:
                    if len(plane.intersection(wall)) > 0:
                        # vertices.append(plane.intersection(wall))
                        room.append(plane)
                        break
                if plane not in room:
                    deleted[i] = True
    if len(pi) == 2:
        for wall1 in range(len(room)):
            for wall2 in range(wall1):
                if wall1 != wall2:
                    intersections = room[wall2].intersection(room[wall1])
                    if len(intersections) > 0:
                        for intersection in intersections:
                            if abs(float(intersection.x)) < 100 and abs(float(intersection.y)) < 100:
                                vertices.append(intersection)

    if len(pi) == 3:
        planes_intersections = []
        for wall1 in range(len(room)):
            for wall2 in range(wall1):
                if wall1 != wall2:
                    planes_intersections.append(room[wall2].intersection(room[wall1]))

        for inter1 in range(len(planes_intersections)):
            for inter2 in range(inter1):
                if inter1 != inter2:
                    intersections = planes_intersections[inter2][0].intersection(planes_intersections[inter1][0])
                    if len(intersections) > 0:
                        for intersection in intersections:
                            if abs(float(intersection.x)) < 100 and abs(float(intersection.y)) < 100 and abs(float(intersection.z)) < 100 and intersection not in vertices:
                                vertices.append(intersection)
    return room, vertices


def plot_room(room, vertices):
    """
    This method...
    :param room:
    :param vertices:
    :return:
    """
    # Plotting in 2D case
    if room[0].ambient_dimension == 2:
        plt.figure()
        ax = plt.axes()

        for i, vertex in enumerate(vertices):
            print('Vertex ', i, ': x=', float(vertex.x), ', y=', float(vertex.y))
            ax.scatter(float(vertex.x), float(vertex.y))
        plt.show()

    else:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        for i, vertex in enumerate(vertices):
            print('Vertex ', i, ': x=', float(vertex.x), ', y=', float(vertex.y), ', z=', float(vertex.z))
            ax.scatter3D(float(vertex.x), float(vertex.y), float(vertex.z))
        plt.show()


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
