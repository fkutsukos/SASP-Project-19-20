import numpy as np
from scipy.optimize import minimize


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
