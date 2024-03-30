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

    def add_bus(self, bus_name: str, bus_voltage, bus_type: str = "none", v_mag = 1, Pg = None):
        if bus_name not in self.buses.keys():
            self.buses[bus_name] = Bus(bus_name, bus_voltage, bus_type, v_mag, Pg)
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

    def calc_mismatch(self, V):
        #calc size of y array
        #assuming one slack find number of PV busses
        iterator = 0
        slack_position: List[int] = list()
        pv_positions: List[int] = list()
        for c in self.buses:
            if self.buses[c].bus_type == "slack":
                slack_position.extend([iterator])
            elif self.buses[c].bus_type == "PV":
                pv_positions.extend([iterator])
            iterator += 1
        size = 2*len(self.buses)-2 - len(pv_positions)
        mismatch = np.zeros(size, dtype=float)
        return

    def calc_solution(self, y, J):
        size = 2*len(self.buses)
        solution = np.zeros(size, dtype=float)

    def calc_j1(self, V):
        size = (len(self.buses) - 1, len(self.buses) - 1)
        d = np.zeros(size)
        J1 = pd.DataFrame(data=d, dtype=float)
        # determine which is slack bus

        slack_position = 0
        for c in self.buses:
            if self.buses[c].bus_type == "slack":
                break
            slack_position += 1
        k = 0
        for a in self.buses:
            if self.buses[a].bus_number == slack_position:
                continue
            n = 0
            for b in self.buses:
                if self.buses[b].bus_number == slack_position:
                    continue
                if k != n:
                    J1.loc[k, n] = (np.abs(V[k]) * np.abs(self.final_y_bus.loc[a, b]) * np.abs(V[n])
                                    * np.sin(np.angle(V[k]) - np.angle(V[n])
                                             - np.angle(self.final_y_bus.loc[a, b])))

                else:
                    temp: float = 0
                    q = 0
                    for p in self.buses:

                        if self.buses[p].bus_number == k:
                            q += 1
                            continue
                        temp += np.abs(self.final_y_bus.loc[a, p]) * np.abs(V[q]) * np.sin(np.angle(V[k]) - np.angle(V[q]) - np.angle(self.final_y_bus.loc[a, p]))
                        q += 1
                    J1.loc[k, n] = -1 * np.abs(V[k]) * temp
                n += 1

            k += 1
        return J1

    def calc_j2(self, V):

        # determine which is slack bus
        iterator = 0
        slack_position: List[int] = list()
        pv_positions: List[int] = list()
        for c in self.buses:
            if self.buses[c].bus_type == "slack":
                slack_position.extend([iterator])
            elif self.buses[c].bus_type == "PV":
                pv_positions.extend([iterator])
            iterator += 1

        size = (len(self.buses) - 1, len(self.buses) - 1 - len(pv_positions))
        d = np.zeros(size)
        J2 = pd.DataFrame(data=d, dtype=float)
        k = 0

        for a in self.buses:
            if self.buses[a].bus_number in slack_position:
                continue
            n = 0
            for b in self.buses:
                if self.buses[b].bus_number in slack_position or self.buses[b].bus_number in pv_positions:
                    continue
                if k != n:
                    J2.loc[k, n] = (np.abs(V[k]) * np.abs(self.final_y_bus.loc[a, b]) * np.abs(V[n])
                                    * np.sin(np.angle(V[k]) - np.angle(V[n])
                                             - np.angle(self.final_y_bus.loc[a, b])))

                else:
                    temp: float = 0
                    q = 0
                    for p in self.buses:

                        if p == k:
                            continue
                        temp += np.abs(self.final_y_bus.loc[a, p]) * np.abs(V[q]) * np.sin(np.angle(V[k])
                                                                                           - np.angle(V[q]) - np.angle(
                            self.final_y_bus.loc[a, p]))
                        q += 1
                    J2.loc[k, n] = -1 * np.abs(V[k]) * temp
                n += 1

            k += 1
        return J2

    def calc_j3(self, V):
        # determine which is slack bus
        iterator = 0
        slack_position: List[int] = list()
        pv_positions: List[int] = list()
        for c in self.buses:
            if self.buses[c].bus_type == "slack":
                slack_position.extend([iterator])
            elif self.buses[c].bus_type == "PV":
                pv_positions.extend([iterator])
            iterator += 1

        size = (len(self.buses) - 1 - len(pv_positions), len(self.buses) - 1)
        d = np.zeros(size)
        J3 = pd.DataFrame(data=d, dtype=float)

        return J3

    def calc_j4(self, V):
        # determine which is slack bus
        iterator = 0
        slack_position: List[int] = list()
        pv_positions: List[int] = list()
        for c in self.buses:
            if self.buses[c].bus_type == "slack":
                slack_position.extend([iterator])
            elif self.buses[c].bus_type == "PV":
                pv_positions.extend([iterator])
            iterator += 1

        size = (len(self.buses) - 1 - len(pv_positions), len(self.buses) - 1 - len(pv_positions))
        d = np.zeros(size)
        J4 = pd.DataFrame(data=d, dtype=float)

        return J4

    def calc_jacobian(self, J1: pd.DataFrame, J2: pd.DataFrame, J3, J4):
        iterator = 0
        slack_position: List[int] = list()
        pv_positions: List[int] = list()
        for c in self.buses:
            if self.buses[c].bus_type == "slack":
                slack_position.extend([iterator])
            elif self.buses[c].bus_type == "PV":
                pv_positions.extend([iterator])
            iterator += 1
        #J_temp = [[J1, J2], [J3, J4]]
        #pd.concat([J1, J2], axis=1, ignore_index=True)
        size = (len(self.buses)-1 + len(self.buses) - 1 - len(pv_positions), len(self.buses)-1 + + len(self.buses) - 1 - len(pv_positions))
        d = np.zeros(size)

        J = pd.DataFrame(data=d, index=range(size[0]), columns=range(size[1]), dtype=float)
        return J




    def simulate(self):
        #initialize bus voltages to 0 except for slack and pv buses
        V = pd.array(data=np.zeros(len(self.buses)), dtype=complex)
        a = 0
        for i in self.buses:

            if self.buses[i].bus_type != "slack" and self.buses[i].bus_type != "PV":
                self.buses[i].set_bus_voltage_mag(1)
            self.buses[i].set_bus_voltage_angle(0)
            V[a] = self.buses[i].v_mag * np.exp(1j*self.buses[i].v_angle)
            a += 1

        y = self.calc_mismatch(V)
        J1 = self.calc_j1(V)
        J2 = self.calc_j2(V)
        J3 = self.calc_j3(V)
        J4 = self.calc_j4(V)
        J = self.calc_jacobian(J1, J2, J3, J4)
        x = self.calc_solution(y, J)





        print("hi")
