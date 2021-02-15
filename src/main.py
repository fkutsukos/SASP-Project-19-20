# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import methods
import peaking_test
import echo_test
import numpy as np
import matplotlib.pyplot as plt
import load_save as ls


if __name__ == '__main__':

    while True:
        print('Provide the experiment name(.json) file')
        user_input = input('> : ')

        try:
            # Load configuration and build room
            room_data = methods.input_data(file_dir="../input", file_name=user_input + '.json')
            rt60 = float(room_data['rt60'])
            dimensions = list(map(float, room_data['dimensions']))
            mic_array = [list(map(float, lst)) for lst in room_data['mic_array']]
            loudspeaker = list(map(float, room_data['source']))
            fs = int(room_data['fs'])
            room, global_delay = methods.build_room(dimensions, loudspeaker, mic_array, rt60, fs)

        except Exception as e:
            print(e)
            continue

        print('Type \'C\' for computing the echoes OR \'R\' for restoring them from a csv file')
        cmd = input('> : ')
        if cmd == 'C':
            try:

                edm = methods.build_edm(np.array(mic_array))

                '''
                Code to implement spline interpolation.
                mic_rirs = []

                for mic in range(len(room.rir)):
                    mic_rirs.append(methods.spline_interpolation(room.rir[mic][0], 100))
                '''
                peaks = peaking_test.find_echoes(room.rir, global_delay=global_delay, n=15).astype(int)

                intermic_max_distance = np.sqrt(np.max(edm))
                # Trying to locate the source considering only one peak per RIR
                echoes = echo_test.echo_sorting(edm, np.array(peaks, dtype=float), len(dimensions), intermic_max_distance, fs, global_delay)
                number_of_microphones = np.array(mic_array).shape[1]
                csv_file_name = user_input + '.csv'
                print('Echoes are saved to ' + csv_file_name)
                ls.save_to_csv(echoes, csv_file_name, number_of_microphones)

            except Exception as e:
                print(e)
                break

        elif cmd == 'R':

            try:
                csv_file_name = user_input+'.csv'
                echoes = ls.load_from_csv(csv_file_name)
            except Exception as e:
                print(e)
                break

        else:
            continue

        virtual_sources = []
        echoes.sort()

        # Computing and printing in 3D the virtual sources
        fig = plt.figure()
        if len(dimensions) == 3:
            ax = plt.axes(projection="3d")
        else:
            ax = plt.axes()

        for echo in echoes:
            source = methods.trilateration(np.array(mic_array), echo[1])
            virtual_sources.append(source)
            if len(dimensions) == 3:
                ax.scatter3D(*source)
            else:
                ax.scatter(*source)
        plt.show()

        # Reconstructing the room using the virtual sources location
        room, vertices = methods.reconstruct_room(virtual_sources, np.array(loudspeaker), 1e-2)
        methods.plot_room(room, vertices)
