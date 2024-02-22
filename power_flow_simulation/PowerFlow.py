from Bus import Bus
from Transformer import Transformer
from Transmission_line import Transmission_line
from typing import Dict, List, Optional
from Settings import s

class PowerFlow:

    def __init__(self, name):
        self.name = name

        # create array of bus name and bus number

        self.buses_order: List[str] = list()
        self.buses: Dict[str, Bus] = dict()

        self.transformers: Dict[str, Transformer] = dict()
        self.transmissionLines: Dict[str, Transmission_line] = dict()


    def add_bus(self, bus_name: str, bus_voltage):
        if bus_name not in self.buses.keys():
            self.buses[bus_name] = Bus(bus_name, bus_voltage)
            self.buses_order.append(bus_name)

    def add_transformer(self, name, power_rating, percent_z, x_r_ratio,  busA: Bus, busB: Bus,):
        # create tx and then automatically calculates parameters
        self.transformers[name] = Transformer(power_rating, percent_z, x_r_ratio, busA, busB)

    def add_transmissionLine(self):


