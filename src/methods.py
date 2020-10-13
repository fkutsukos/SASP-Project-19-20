import numpy as np
import pyroomacoustics as pra


def build_room(dimensions, source, mic_array, rt60 ,fs):
    """
    This method wraps inside all the necessary steps
    to build the simulated room
    :param dimensions: length, width and height of the room
    :param source: contains the x, y, z location of the sound source
    :param mic_array: contains the x, y, z vectors of all the microphones
    :param float rt60: represents the reverberation time of the room
    :param int fs: represents the sampling frequency used for the generation of signals
    :return: pyroomacoustics object representing the room
    """
    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60, dimensions)

    # Building a 'Shoebox' room with the provided dimensions
    room = pra.ShoeBox(p=dimensions, fs=fs, absorption=e_absorption, max_order= max_order)

    # Place the Microphone Array and the Sound Source inside the room
    mics = pra.MicrophoneArray(mic_array, fs)
    room.add_microphone_array(mics)
    room.add_source(source)

    # Computing the Room Impulse Response at each microphone, for each source
    room.image_source_model()
    room.compute_rir()

    return room


def peakPicking(mic_rir, numberOfEchoes):
    return peaksArray


def buildEDM(R):
    EDM = np.array([[0, 2], [2, 0]])
    return EDM


def echoLabeling(EDM,peaksArray):
    peaks = np.array([[20, 25, 30, 35],
                          [25, 30, 36, 40]])
    c = 340 # speed of sound
    Fs = 16000 # sampling freq
    peaks *= c/Fs
    #for each row of peaks
        #for each column of peaks
            #rankTest
            #if rankTest == True
             #   store combination of distances/image sources
             #   array indexes of echoes 1st image source [20 , 30] , 2nd [30 , 36] , 3rd [35, 40] 4th ...

             #   break;


    return peaksToImageSourceArray


def trilateration(sortedEchoestoImageSources):
    # using Linear Least Squares
    # using System of Linear Equations
