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

# Add a sound source
room.add_source([1.5, 1.2, 1.6])

# Place a linear 2D array in the room
center = np.array([np.floor(length / 2), np.floor(width / 2), np.floor(height / 2)])
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

# Run image source model and show room and 2nd order images

room.image_source_model()
fig = plt.figure("Rirs")
room.plot_rir()
room.plot(img_order=1, aspect='equal')
plt.show()
