# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import methods
import numpy as np
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Testing the Trilateration method with the data contained in the paper Murphy-Hereman Trilateration
    """
    source = methods.trilateration(np.array([[475060, 481500, 482230, 478050, 471430, 468720, 467400, 468730],
                                             [1096300, 1094900, 1088430, 1087810, 1088580, 1091240, 1093980, 1097340],
                                             [4670, 4694, 4831, 4775, 4752, 4803, 4705, 4747]]),
                                   np.array([5942.607, 2426.635, 5094.254, 5549.874, 9645.353, 11419.870, 12639.330,
                                             12078.820]))
    print(source)
    """
    # testing the build room using input from json file
    room_data = methods.input_data(file_dir="../input", file_name="room.json")
    rt60 = float(room_data['rt60'])
    dimensions = list(map(float, room_data['dimensions']))
    mic_array = [list(map(float, lst)) for lst in room_data['mic_array']]
    source = list(map(float, room_data['source']))
    fs = int(room_data['fs'])

    room, global_delay = methods.build_room(dimensions, source, mic_array, rt60, fs)
    edm = methods.build_edm(np.array(mic_array))
    peaks = methods.peak_picking(room.rir, 5)
    echoes = methods.echo_labeling(edm, np.array(peaks, dtype=float), 3, fs, global_delay)

    virtual_sources = []

    for echo in echoes:
        virtual_sources.append(methods.trilateration(np.array(mic_array), echo))