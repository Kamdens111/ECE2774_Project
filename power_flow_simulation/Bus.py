import numpy as np

# number of busses
# bus name
# create power system class

class Bus:
    bus_count = 0

    def __init__(self, name, voltage_base):
        self.voltage_base = voltage_base
        self.name = name
        self.number = Bus.bus_count

        self.v = None
        Bus.bus_count += 1

    def set_bus_voltage(self, bus_v):
        self.v=bus_v

    def show_bus_count(self):
        print(self.bus_count)
