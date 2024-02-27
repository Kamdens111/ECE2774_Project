import numpy as np

# number of busses
# bus name
# create power system class

class Bus:
    bus_count = 0

    def __init__(self, name, nominal_voltage):
        self.voltage_base = nominal_voltage
        self.name = name
        self.bus_number = Bus.bus_count

        self.v = None
        Bus.bus_count += 1

    def set_bus_voltage(self, bus_v):
        self.v = bus_v

    def show_bus_number(self):
        print(self.bus_number)

    def show_bus_count(self):
        print(Bus.bus_count)
