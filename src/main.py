# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import methods
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Testing the Trilateration method with the data contained in the paper Murphy-Hereman Trilateration
    source =methods.trilateration(np.array([[475060, 481500, 482230, 478050, 471430, 468720, 467400, 468730],
                                            [1096300, 1094900, 1088430, 1087810, 1088580, 1091240, 1093980, 1097340],
                                            [4670, 4694, 4831, 4775, 4752, 4803, 4705, 4747]]),
                                  np.array([5942.607, 2426.635, 5094.254, 5549.874, 9645.353, 11419.870, 12639.330, 12078.820]))
    print(source)
    print_hi("SASP")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
