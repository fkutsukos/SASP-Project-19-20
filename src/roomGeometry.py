import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# The desired reverberation time and dimensions of the room
rt60 = 1.0  # seconds
length = 15.0  # meters
width = 10.0
height = 3.0
room_dim = [length, width, height]

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

# Create a shoebox room
room = pra.ShoeBox(
    room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order
)

# Place a linear 2D array in the room
center = np.array([(length / 2), (width / 2), (height / 2)])

d = 1  # distance between array
x1 = center[0]
x2 = x1 + d
x3 = x2 + d
x4 = x1 - d
x5 = x4 - d
y = center[1]
z = center[2]
R = np.array([[x1, x2, x3, x4, x5], [y, y, y, y, y], [z, z, z, z, z]])
bf = pra.MicrophoneArray(R, room.fs)
room.add_microphone_array(bf)

mic = room.mic_array.R

# Add a sound source
room.add_source([center[0], center[1] + 2, center[2]])

# Run image source model and show room and 2nd order images
room.image_source_model()
room.compute_rir()

# Plot the Rirs and the Room
#room.plot_rir()
# room.plot(img_order=1, aspect='equal')
#plt.show()


#peak localization
from scipy.signal import find_peaks
rir1 = np.asarray(room.rir[1])
rir1 = np.transpose(rir1).flatten()
peaks, _ = find_peaks(rir1,distance=5, threshold=0.05, height=0.03);
#plt.plot(rir1)
#plt.plot(peaks, rir1[peaks], "x")
#plt.plot(np.zeros_like(rir1), "--", color="gray")
#plt.show()

#Build EDM matrix



# store the Rirs into a matrix
#rirs_orig = room.rir
#rirs = np.array([])
#for row in rirs_orig:
    #rirs = np.append(rirs, np.array(row))

#rirs = np.array(np.asarray(room.rir))

print("end")

