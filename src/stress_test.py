import numpy as np
from sklearn.manifold import MDS
from scipy.optimize import minimize


daug = np.array([[0.0, 11.0, 2.0, 10.0, 2.0, 161.111],
                     [11.0, 0.0, 5.0, 5.0, 9.0, 178.778],
                     [2.0, 5.0, 0.0, 8.0, 4.0, 149.778],
                     [10.0, 5.0, 8.0, 0.0, 4.0, 224.444],
                     [2.0, 9.0, 4.0, 4.0, 0.0, 196.444],
                     [161.111, 178.778, 149.778, 224.444, 196.444, 0.0]])


def stress(x, daug):
    m = len(daug)
    X = x.reshape((-1, m))
    stress_score = 0
    for j in range(m):
        for i in range(m):
            stress_score = stress_score + (np.linalg.norm(X[:, j] - X[:, i])**2 - daug[i, j])**2

    return stress_score


res = minimize(stress, x0=np.random.rand(1, len(daug)*3), args=(daug,), method='SLSQP', options={'disp':True})


embedding = MDS(n_components=3, dissimilarity="precomputed")
embedding.fit(daug)

print(embedding.embedding_)
print(embedding.stress_)

