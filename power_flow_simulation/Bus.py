import numpy as np

# number of busses
# bus name
# create power system class


class Bus:

    bus_count = 0
    slack_bus_count = 0

    def __init__(self, name, nominal_voltage, bus_type: str = "PQ", v_mag: float = 1, Pg: float = None):
        self.voltage_base = nominal_voltage
        self.name = name
        self.bus_number = Bus.bus_count
        self.bus_type = bus_type
        self.v_mag = v_mag
        self.P_inj = Pg

        #self.v_mag = None
        #self.v_angle = None
        Bus.bus_count += 1

        if bus_type == "slack":
            self.v_mag = 1
            self.v_angle = 0
            Bus.slack_bus_count += 1
        if Bus.slack_bus_count >= 2:
            exit("ERROR! More than one slack bus")


    def set_bus_voltage_mag(self, bus_v):
        self.v_mag = bus_v

    def set_bus_voltage_angle(self, bus_angle):
        self.v_angle = bus_angle

    def show_bus_number(self):
        print(self.bus_number)

    def get_bus_count(self):
        return Bus.bus_count
