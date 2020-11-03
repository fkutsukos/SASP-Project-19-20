import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.signal import find_peaks

import methods

# TODO: Create User Input for rt60, room_dimensions, central_position, mic_distance, fs
# The desired reverberation time and dimensions of the room
rt60 = 1.0  # seconds
length = 15.0  # meters
width = 10.0
height = 3.0
room_dim = [length, width, height]
center = np.array([(length / 2), (width / 2), (height / 2)])
# Create the R (numpy.ndarray) – Mics positions
mic_distance = 1  # distance between microphones in the array
x = center[0]
y = center[1]
z = center[2]
R = np.array([[x, x + mic_distance, x + 2 * mic_distance, x - mic_distance, x - 2 * mic_distance],
              [y, y, y, y, y],
              [z, z, z, z, z]])
# Create the source
source = np.array([center[0], center[1] + 2, center[2]])
# TODO: end of user input

room = methods.build_room(dimensions=room_dim, source=source, mic_array=R, rt60=rt60, fs=16000)

"""
# Plot the Rirs and the Room
room.plot_rir()
room.plot(img_order=1, aspect='equal')
plt.show()
"""

# peak localization - A good strategy is to select a number of peaks greater than the number of walls
numberOfWalls = len(room_dim)*2

#peaksArray = methods.peakPicking(mic_rir=room, numberOfEchoes=numberOfWalls**2)

rir1 = np.asarray(room.rir[1])
rir1 = np.transpose(rir1).flatten()
peaks, _ = find_peaks(rir1,distance=5, threshold=0.05, height=0.03)
plt.plot(rir1)
plt.plot(peaks, rir1[peaks], "x")
plt.plot(np.zeros_like(rir1), "--", color="gray")
plt.show()

# Echo Labeling -  Build EDM matrix

# store the Rirs into a matrix
# rirs_orig = room.rir
# rirs = np.array([])
# for row in rirs_orig:
# rirs = np.append(rirs, np.array(row))

# rirs = np.array(np.asarray(room.rir))

print("end")
