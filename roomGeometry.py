import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra


# Create a 2D room from the corners of a polygon
floor = np.array([[0, 0, 10, 10],   # x−coordinates
                  [0, 10, 10, 0]])  # y−coordinates
room = pra.Room.from_corners(floor, fs=16000, max_order=12, absorption=0.1)
# Lift the room in 3D space
room.extrude(2.4)
# Add a sound source
room.add_source([1.5, 1.2, 1.6])
# Place two microphones in the room
R = np.array([[3., 4.2], [2.25, 2.1], [1.4, 1.4]])
bf = pra.MicrophoneArray(R, room.fs)
room.add_microphone_array(bf)
# Run image source model and show room and 2nd order images
room.image_source_model()
room.plot(img_order=3, aspect='equal')

fig = plt.figure()
room.plot_rir()
plt.show()
