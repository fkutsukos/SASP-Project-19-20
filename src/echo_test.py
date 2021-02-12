import numpy as np
from scipy.optimize import minimize


def echo_sorting(edm, mic_peaks, k, diameter, fs, global_delay, c=343.0):
    neighbour_samples = (diameter / c) * fs

    T1 = mic_peaks[0]

    sorted_echoes = []

    for t1 in T1:
        local_peaks = []
        for echoes in mic_peaks[1:]:
            echo = [echo for echo in echoes if t1 - neighbour_samples <= echo <= t1 + neighbour_samples]
            if len(echo) > 0:
                echo = np.array(echo)
                echo -= global_delay
                echo *= c / fs
                local_peaks.append(echo)

        if len(local_peaks) == len(mic_peaks) - 1:
            t1 -= global_delay
            t1 *= c / fs

            echoes_comb = np.array(np.meshgrid(*local_peaks)).T.reshape(-1, len(local_peaks))

            dim = edm.shape[0]
            dim += 1

            d_aug = np.zeros(shape=(dim, dim))

            d_aug[0:dim - 1, 0:dim - 1] = edm

            score_best = np.inf
            d_best = []

            for echo in echoes_comb:
                d = np.insert(echo, 0, t1, axis=0)

                d_aug[0:-1, -1] = d ** 2
                d_aug[-1, 0:-1] = d ** 2

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

                score = res.fun

                if score <= score_best:
                    score_best = score
                    d_best = d

            sorted_echoes.append((score_best, d_best))

    return sorted_echoes
