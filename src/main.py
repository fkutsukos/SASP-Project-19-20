# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import methods, peaking_test, echo_test
import numpy as np
import matplotlib.pyplot as plt
import load_save as ls



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
    compute = False
    restore = False

    while True:
        print('Type \'C\' for computing the echoes OR \'R\' for restoring them')
        try:
            cmd = input('> : ')

        except KeyboardInterrupt:
            print('Exiting')
            break

        try:
            # Testing the build room using input from json file
            configuration_file = "room.json"
            csv_file_name = configuration_file.split('.')[0] + '.csv'

            room_data = methods.input_data(file_dir="../input", file_name=configuration_file)
            rt60 = float(room_data['rt60'])
            dimensions = list(map(float, room_data['dimensions']))
            mic_array = [list(map(float, lst)) for lst in room_data['mic_array']]
            loudspeaker = list(map(float, room_data['source']))
            fs = int(room_data['fs'])

            room, global_delay = methods.build_room(dimensions, loudspeaker, mic_array, rt60, fs)

        except Exception as e:
            print(e)

        if cmd == 'C':
            print('Computing the echoes..')
            try:
                edm = methods.build_edm(np.array(mic_array))
                '''
                Code to implement spline interpolation.
                mic_rirs = []
            
                for mic in range(len(room.rir)):
                    mic_rirs.append(methods.spline_interpolation(room.rir[mic][0], 100))
                '''
                peaks = peaking_test.find_echoes(room.rir, global_delay=global_delay, n=5).astype(int)

                # Trying to locate the source considering only one peak per RIR
                # peaks = methods.peak_picking(room.rir, 3)

                #echoes = methods.echo_labeling(edm, np.array(peaks, dtype=float), 2, fs, global_delay)
                echoes = echo_test.echo_sorting(edm, np.array(peaks, dtype=float), 2, 4.0, fs, global_delay)


                csv_file_name = configuration_file.split('.')[0] + '.csv'
                number_of_microphones = np.array(mic_array).shape[1]
                ls.save_to_csv(echoes, csv_file_name, number_of_microphones)
                print('echoes are saved to ' + csv_file_name)

            except Exception as e:
                print(e)

        if cmd == 'R':
            try:
                print('echoes are laoded from ' + csv_file_name)
                echoes = ls.load_from_csv(csv_file_name)
            except Exception as e:
                print(e)
        try:
            virtual_sources = []
            echoes.sort()
            # Computing and printing in 3D the virtual sources
            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = plt.axes()
            for echo in echoes:
                # if echo[0] < np.inf:
                # source = methods.trilaterate_beck(np.array(mic_array).transpose(), echo[1])
                source = methods.trilateration(np.array(mic_array), echo[1])
                virtual_sources.append(source)
                # ax.scatter3D(*source)
                ax.scatter(*source)
                print('scattered source no:' , len(virtual_sources) )
            plt.show()
            print(0)
            methods.reconstruct_room(virtual_sources, np.array(loudspeaker), 1e-5)

        except Exception as e:
            print(e)

        break
