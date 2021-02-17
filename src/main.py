import room_methods
import peaks_methods
import echoes_methods
import numpy as np
import matplotlib.pyplot as plt
import load_save as ls

NUM_ECHOES = 15
MAX_ORDER = 2
DIST_THRESH = 1e-1

if __name__ == '__main__':

    while True:
        print('Provide the experiment name(.json) file')
        user_input = input('> : ')

        try:
            # Load configuration and build room
            room_data = room_methods.input_data(file_dir="../input", file_name=user_input + '.json')
            rt60 = float(room_data['rt60'])
            dimensions = list(map(float, room_data['dimensions']))
            mic_array = [list(map(float, lst)) for lst in room_data['mic_array']]
            loudspeaker = list(map(float, room_data['source']))
            fs = int(room_data['fs'])
            room, global_delay = room_methods.build_room(dimensions, loudspeaker, mic_array, rt60, fs, MAX_ORDER)

        except Exception as e:
            print(e)
            continue

        print('Type \'C\' for computing the echoes OR \'R\' for restoring them from a csv file')
        cmd = input('> : ')
        if cmd == 'C':
            try:

                edm = echoes_methods.build_edm(np.array(mic_array))

                peaks = peaks_methods.find_echoes(room.rir, n=NUM_ECHOES).astype(int)

                # Printing the RIR with the identified peaks.
                rir1 = room.rir[1][0]
                plt.plot(rir1)
                plt.plot(peaks[1], rir1[peaks[1]], "x")
                plt.plot(np.zeros_like(rir1), "--", color="gray")
                plt.show()

                inter_mic_max_distance = np.sqrt(np.max(edm))

                echoes = echoes_methods.echo_sorting(edm, np.array(peaks, dtype=float), len(dimensions),
                                                     inter_mic_max_distance, fs, global_delay)

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
            source = echoes_methods.trilateration(np.array(mic_array), echo[1])
            virtual_sources.append(source)
            if len(dimensions) == 3:
                ax.scatter3D(*source)
            else:
                ax.scatter(*source)
        plt.show()

        # Reconstructing the room using the virtual sources location
        room, vertices = room_methods.reconstruct_room(virtual_sources, np.array(loudspeaker), dist_thresh=DIST_THRESH)
        room_methods.plot_room(room, vertices)
