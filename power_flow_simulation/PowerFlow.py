import pandas as pd
import numpy as np

from Bus import Bus
from Transformer import Transformer
from TransmissionLine import TransmissionLine
from Geometry import Geometry
from Conductor import Conductor
from typing import Dict, List, Optional
from Settings import s


class PowerFlow:

    def __init__(self, name):
        self.name = name

        # create array of bus name and bus number

        self.buses_order: List[str] = list()
        self.buses: Dict[str, Bus] = dict()

        self.transformers: Dict[str, Transformer] = dict()
        self.transmissionLines: Dict[str, TransmissionLine] = dict()
        self.geometry: Dict[str, Geometry] = dict()
        self.conductors: Dict[str, Conductor] = dict()

    def add_bus(self, bus_name: str, bus_voltage):
        if bus_name not in self.buses.keys():
            self.buses[bus_name] = Bus(bus_name, bus_voltage)
            self.buses_order.append(bus_name)

    def add_transformer(self, name, power_rating, percent_z, x_r_ratio,  busA: str, busB: str,):
        # create tx and then automatically calculates parameters
        self.transformers[name] = Transformer(power_rating, percent_z, x_r_ratio, self.buses[busA], self.buses[busB])

    def add_geometry(self, name, d_ab, d_bc, d_ca, conductor_distance, num_conductors):
        self.geometry[name] = Geometry(d_ab, d_bc, d_ca, conductor_distance, num_conductors)

    def add_conductor(self, name, GMR, R_c, d_conductor, geometry_name: str):
        self.conductors[name] = Conductor(GMR, R_c, d_conductor, self.geometry[geometry_name])

    def add_transmissionLine(self, name, line_length, conductor_name: str, busA: str, busB: str):
        self.transmissionLines[name] = TransmissionLine(line_length, self.conductors[conductor_name], self.buses[busA],
                                                        self.buses[busB])

    def get_y_bus(self):
        size = (len(self.buses), len(self.buses))
        d = np.zeros(size)
        self.final_y_bus = pd.DataFrame(data=d, index=self.buses_order, columns=self.buses_order, dtype=complex)
        # add transformer ybuses
        for x in self.transformers.keys():
            self.final_y_bus.loc[self.transformers[x].busA.name, self.transformers[x].busA.name] += (
                self.transformers[x].y_bus.loc)[self.transformers[x].busA.name, self.transformers[x].busA.name]
            self.final_y_bus.loc[self.transformers[x].busB.name, self.transformers[x].busB.name] += (
                self.transformers[x].y_bus.loc)[self.transformers[x].busB.name, self.transformers[x].busB.name]
            self.final_y_bus.loc[self.transformers[x].busA.name, self.transformers[x].busB.name] += (
                self.transformers[x].y_bus.loc)[self.transformers[x].busA.name, self.transformers[x].busB.name]
            self.final_y_bus.loc[self.transformers[x].busB.name, self.transformers[x].busA.name] += (
                self.transformers[x].y_bus.loc)[self.transformers[x].busB.name, self.transformers[x].busA.name]

        #add tline ybuses
        for x in self.transmissionLines.keys():
            self.final_y_bus.loc[self.transmissionLines[x].busA.name, self.transmissionLines[x].busA.name] += (
                self.transmissionLines[x].y_bus.loc)[self.transmissionLines[x].busA.name, self.transmissionLines[x].busA.name]
            self.final_y_bus.loc[self.transmissionLines[x].busB.name, self.transmissionLines[x].busB.name] += (
                self.transmissionLines[x].y_bus.loc)[self.transmissionLines[x].busB.name, self.transmissionLines[x].busB.name]
            self.final_y_bus.loc[self.transmissionLines[x].busA.name, self.transmissionLines[x].busB.name] += (
                self.transmissionLines[x].y_bus.loc)[self.transmissionLines[x].busA.name, self.transmissionLines[x].busB.name]
            self.final_y_bus.loc[self.transmissionLines[x].busB.name, self.transmissionLines[x].busA.name] += (
                self.transmissionLines[x].y_bus.loc)[self.transmissionLines[x].busB.name, self.transmissionLines[x].busA.name]

        return self.final_y_bus

