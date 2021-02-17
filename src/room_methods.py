import numpy as np
import sympy as sp
import pyroomacoustics as pra
import json
import os
import matplotlib.pyplot as plt

# noinspection PyDeprecation


def build_room(dimensions, source, mic_array, rt60, fs, max_order):
    """
    This method wraps inside all the necessary steps
    to build the simulated room
    :param dimensions: length, width and height of the room
    :param source: contains the x, y, z location of the sound source
    :param mic_array: contains the x, y, z vectors of all the microphones
    :param float rt60: represents the reverberation time of the room
    :param int fs: represents the sampling frequency used for the generation of signals
    :param int max_order: represents the maximum order of the simulated reflections
    :return: pyroomacoustics object representing the room and the fractional delay to compensate
    """
    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, _ = pra.inverse_sabine(rt60, dimensions)

    # Building a 'Shoebox' room with the provided dimensions
    room = pra.ShoeBox(p=dimensions, fs=fs, absorption=e_absorption, max_order=max_order)

    # Place the Microphone Array and the Sound Source inside the room
    mics = pra.MicrophoneArray(mic_array, fs)
    room.add_microphone_array(mics)
    room.add_source(source)

    # Computing the Room Impulse Response at each microphone, for each source
    room.image_source_model()
    room.compute_rir()

    # Getting the fractional delay introduced by the simulation
    global_delay = pra.constants.get("frac_delay_length") // 2

    # room.plot()

    return room, global_delay


def input_data(file_dir="../input", file_name="room.json"):
    """
    This method receives the input data from a JSON formatted file
    :param: file_dir: the directory from where to read the data
    :param: file_name: the name of the file to be read
    :return: the dictionary of input data
    """
    data = {}
    path_to_file = os.path.join(file_dir, file_name)
    if os.path.exists(path_to_file):
        with open(path_to_file) as f:
            data = json.load(f)
    else:
        print('File not found in directory {}'.format(file_dir))
    return data


def reconstruct_room(candidate_virtual_sources, loudspeaker, dist_thresh, shoebox=True):
    """
    This method uses the first-order virtual-sources to reconstruct the room:
    it processes the candidate virtual sources in the order of increasing distance
    from the loudspeaker to find the first-order virtual sources and add their planes
    to the list of half-spaces whose intersection determines the final room.
    :param candidate_virtual_sources: list of the coordinates of all the individuated
    virtual sources (it could contain even higher-order virtual sources)
    :param loudspeaker: x, y, z coordinates of the speaker location in the room
    :param dist_thresh: distance threshold (epsilon)
    :param shoebox: boolean to identify if the room is a shoebox
    :return: list of planes corresponding to the first-order virtual sources
    """

    def combine(s1, s2):
        """
        This method combines the virtual sources s1 and s2 to generate a higher-order
        virtual source; it is used as a criterion to discard higher-order virtual sources.
        :param s1: first virtual source coordinates
        :param s2: second virtual source coordinates
        :return: the coordinates of the higher order virtual source generated through
        the combination of s1 and s2
        """
        # p2 is a point on the hypothetical wall defined by s2, that is, a point on
        # the median plane between the loudspeaker and s2
        p2 = (loudspeaker + s2) / 2

        # n2 is the outward pointing unit normal
        n2 = (loudspeaker - s2) / np.linalg.norm(loudspeaker - s2)

        return s1 + 2 * np.dot((p2 - s1), n2) * n2

    def perpendicular(a):
        """
        This method computes the perpendicular to a given vector.
        :param a: the given vector
        :return: the perpendicular vector
        """
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    # Instantiating the array to contain the distance of each virtual source from the loudspeaker
    distances_from_speaker = []

    # Computing the distances
    for source in candidate_virtual_sources:
        distances_from_speaker.append(np.linalg.norm(source - loudspeaker))

    # Re-ordering the list of virtual sources according to their distance from the loudspeaker
    candidate_virtual_sources = np.array(candidate_virtual_sources)
    sorted_virtual_sources = candidate_virtual_sources[np.array(distances_from_speaker).argsort()][1:]

    # Initialize the list of planes that constitutes the room
    room = []
    vertices = []
    # Initialize the boolean mask to identify the first-order virtual sources
    deleted = np.array([False] * len(sorted_virtual_sources), dtype=bool)

    for i in range(len(sorted_virtual_sources)):
        for j in range(i):
            for k in range(i):
                # The following two conditions verify if the current virtual source is a combination of lower order
                # virtual sources: if so, it is deleted from the available candidates
                if j != k and k < i:
                    if np.linalg.norm(
                            combine(
                                sorted_virtual_sources[j], sorted_virtual_sources[k]
                            ) - sorted_virtual_sources[i]
                    ) < dist_thresh:
                        deleted[i] = True

    # If the room is a "Shoebox" it is possible to exploit geometric properties to filter out
    # "ghost" image sources from the first order reflections.
    if shoebox:

        # Array containing the direction of the axes passing through the source position
        directions = []

        if len(loudspeaker) == 2:
            x = sp.Line(loudspeaker, loudspeaker + [0, 1])
            y = sp.Line(loudspeaker, loudspeaker + [1, 0])
            directions.append(x)
            directions.append(y)

        elif len(loudspeaker) == 3:
            planes = []
            x = sp.Plane(loudspeaker, [1, 0, 0])
            y = sp.Plane(loudspeaker, [0, 1, 0])
            z = sp.Plane(loudspeaker, [0, 0, 1])
            planes.append(x)
            planes.append(y)
            planes.append(z)
            for i in range(3):
                for j in range(i):
                    directions.append(planes[i].intersection(planes[j])[0])

        for i in range(len(sorted_virtual_sources)):
            if not deleted[i]:
                for index, direction in enumerate(directions):
                    if direction.distance(sp.Point(sorted_virtual_sources[i])) < dist_thresh:
                        break
                    else:
                        if index == len(directions) - 1:
                            deleted[i] = True

    # If the virtual source is not a combination of lower order virtual sources, the corresponding plane
    # is built and it is added to the room's walls list

    for i in range(len(sorted_virtual_sources)):
        if not deleted[i]:
            # pi is a point on the hypothetical wall defined by si, that is, a point on
            # the median plane between the loudspeaker and si
            pi = (loudspeaker + sorted_virtual_sources[i]) / 2
            # ni is the outward pointing unit normal
            ni = (loudspeaker - sorted_virtual_sources[i]) / np.linalg.norm(
                loudspeaker - sorted_virtual_sources[i])

            plane = {}

            if len(pi) == 2:
                ni_perp = perpendicular(ni)
                pi2 = pi + ni_perp
                plane = sp.Line(pi, pi2)

            elif len(pi) == 3:
                plane = sp.Plane(sp.Point3D(pi), normal_vector=ni)

            # If the room is empty, we add the first plane to the list of half-spaces whose intersection
            # determines the final room
            if len(room) == 0:
                room.append(plane)
            else:
                for wall in room:
                    if len(plane.intersection(wall)) > 0:
                        room.append(plane)
                        break
                if plane not in room:
                    deleted[i] = True
    if room[0].ambient_dimension == 2:
        for wall1 in range(len(room)):
            for wall2 in range(wall1):
                if wall1 != wall2:
                    intersections = room[wall2].intersection(room[wall1])
                    if len(intersections) > 0:
                        for intersection in intersections:
                            if abs(float(intersection.x)) < 100 and abs(float(intersection.y)) < 100:
                                vertices.append(intersection)

    if room[0].ambient_dimension == 3:
        planes_intersections = []
        for wall1 in range(len(room)):
            for wall2 in range(wall1):
                if wall1 != wall2:
                    planes_intersections.append(room[wall2].intersection(room[wall1]))

        for inter1 in range(len(planes_intersections)):
            for inter2 in range(inter1):
                if inter1 != inter2:
                    intersections = planes_intersections[inter2][0].intersection(planes_intersections[inter1][0])
                    if len(intersections) > 0:
                        for intersection in intersections:
                            if abs(float(intersection.x)) < 100 and abs(float(intersection.y)) < 100 and abs(float(intersection.z)) < 100 and intersection not in vertices:
                                vertices.append(intersection)
    return room, vertices


def plot_room(room, vertices):
    """
    This method takes the room walls and the vertices and plots them
    :param room: list of Sympy Planes or Lines
    :param vertices: list of Points
    """
    # Plotting in 2D case
    if room[0].ambient_dimension == 2:
        ax = plt.axes()

        for i, vertex in enumerate(vertices):
            print('Vertex ', i, ': x=', float(vertex.x), ', y=', float(vertex.y))
            ax.scatter(float(vertex.x), float(vertex.y))
        plt.show()

    else:
        ax = plt.axes(projection="3d")
        for i, vertex in enumerate(vertices):
            print('Vertex ', i, ': x=', float(vertex.x), ', y=', float(vertex.y), ', z=', float(vertex.z))
            ax.scatter3D(float(vertex.x), float(vertex.y), float(vertex.z))
        plt.show()
