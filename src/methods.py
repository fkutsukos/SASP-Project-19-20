import numpy as np


def buildRoom(dimensions, source, mic_array):
    """
    This method wraps inside all the necessary steps
    to build the simulated room
    :param array dimensions: length, width and height of the room
    :param array source: contains the x, y, z location of the sound source
    :param array mic_array: contains the x, y, z location of all the microphones
    :return: pyroomacoustics object representing the room
    """
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
