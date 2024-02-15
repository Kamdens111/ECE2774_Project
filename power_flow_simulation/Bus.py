import numpy as np

# number of busses
# bus name
# create power system class

global bus_count
bus_count = 0

class Bus:
    def __init__(self, number, name, voltage):
        self.voltage = voltage
        self.number = number
        self.name = name
        bus_count += 1
