import numpy as np
from scipy.optimize import minimize


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


def echo_sorting(edm, mic_peaks, k, diameter, fs, global_delay, alpha=0.0,  c=343.0):
    """
    This method evaluates if the matrix Daug, generated adding a row and
    a column made of the possible echo combinations squared to the
    EDM matrix built over the microphone distances is still an
    Euclidean Distance Matrix; this is performed through MDS, assessing
    s-stress function, which defines is used to rank the candidate image
    sources.
    :param edm: Euclidean Distance Matrix built using the distances
    between the different microphones.
    :param mic_peaks: list of arrays containing the indexes of the
    peaks found in the RIR captured by each microphone.
    :param k: dimensionality of the space.
    :param diameter: maximum distance between a couple of microphones,
    used to limit the possible echo combinations.
    :param fs: sampling frequency used in the RIRs measure.
    :param global_delay: fractional delay to be compensated.
    :param alpha: regularization term for distance in stress-score
    :param c: speed of sound (set to the default value of 343 m/s).
    :return: a list containing all the echo combinations that better fit
    the EDM property
    """
    # Computing the number of samples corresponding to the diameter
    neighbour_samples = (diameter / c) * fs

    # Echoes used as reference
    T1 = mic_peaks[0]

    # List which will contain the best scoring set of echoes
    sorted_echoes = []

    # For each echo in the reference echoes
    for t1 in T1:
        local_peaks = []
        for echoes in mic_peaks[1:]:
            echo = [echo for echo in echoes if t1 - neighbour_samples <= echo <= t1 + neighbour_samples]
            if len(echo) > 0:
                echo = np.array(echo)
                # Converting the RIR peak location indexes into space distances
                echo -= global_delay
                echo *= c / fs

                local_peaks.append(echo)

        if len(local_peaks) == len(mic_peaks) - 1:
            # Converting the reference peak location index into space distance
            t1 -= global_delay
            t1 *= c / fs

            # Building a grid of each possible echo combination
            echoes_comb = np.array(np.meshgrid(*local_peaks)).T.reshape(-1, len(local_peaks))

            # Getting the input EDM dimensions and setting Daug dimensions
            dim = edm.shape[0]
            dim += 1

            # Instantiating the Daug matrix
            d_aug = np.zeros(shape=(dim, dim))

            # Start building Daug inserting the initial EDM
            d_aug[0:dim - 1, 0:dim - 1] = edm

            # Instantiating the variable to contain the best scoring echo combination
            score_best = np.inf
            d_best = []

            # Echo labeling process
            for echo in echoes_comb:
                # Building the echo combination in the neighbourhood of the reference peak
                d = np.insert(echo, 0, t1, axis=0)

                # Adding the echo combination squared
                d_aug[0:-1, -1] = d ** 2
                d_aug[-1, 0:-1] = d ** 2

                # s-stress Score Function
                def stress(x, daug):
                    m = len(daug)
                    X = x.reshape((-1, m))
                    stress_score = 0
                    for j in range(m):
                        for i in range(m):
                            # Definition of the s-stress score function
                            stress_score = stress_score + (np.linalg.norm(X[:, j] - X[:, i]) ** 2 - daug[i, j]) ** 2
                        stress_score = stress_score + alpha * np.sqrt(daug[-1, j])
                    return stress_score

                # Multi-Dimensional Scaling using the s-stress function as target
                res = minimize(stress, x0=np.random.rand(1, len(d_aug) * k), args=(d_aug,), method='SLSQP',
                               options={'disp': False})

                # Extracting the optimal function value
                score = res.fun

                # Update the best scoring echo combination
                if score <= score_best:
                    score_best = score
                    d_best = d

            # Update the list of best scoring echoes
            sorted_echoes.append((score_best, d_best))

    return sorted_echoes


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
