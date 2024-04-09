import pandas as pd
import numpy as np

from Generator import Generator
from Load import Load
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
        self.generators: Dict[str, Generator] = dict()
        self.loads: Dict[str, Load] = dict()

    def add_bus(self, bus_name: str, bus_voltage, bus_type: str = "none", v_mag=1):
        if bus_name not in self.buses.keys():
            self.buses[bus_name] = Bus(bus_name, bus_voltage, bus_type, v_mag)
            self.buses_order.append(bus_name)

    def add_transformer(self, name, power_rating, percent_z, x_r_ratio,  busA: str, busB: str,):
        # create tx and then automatically calculates parameters
        self.transformers[name] = Transformer(power_rating, percent_z, x_r_ratio, self.buses[busA], self.buses[busB])

    def add_geometry(self, name, d_ab, d_bc, d_ca, conductor_distance, num_conductors):
        self.geometry[name] = Geometry(d_ab, d_bc, d_ca, conductor_distance, num_conductors)

    def add_conductor(self, name, GMR, R_c, d_conductor, current_limit, geometry_name: str):
        self.conductors[name] = Conductor(GMR, R_c, d_conductor, current_limit, self.geometry[geometry_name])

    def add_transmissionLine(self, name, line_length, conductor_name: str, busA: str, busB: str):
        self.transmissionLines[name] = TransmissionLine(name, line_length, self.conductors[conductor_name], self.buses[busA],
                                                        self.buses[busB])

    def add_generator(self, name: str, bus: str, P=s.S_mva):
        self.generators[name] = Generator(name, bus, P)

    def add_load(self, name: str, bus: str, P, Q):
        self.loads[name] = Load(name, bus, P, Q)

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
        #make voltage vector with [ v delta ]
        full_voltage = np.zeros(2*len(V))
        iterator = 0
        for x in self.buses:
            full_voltage[iterator] = self.buses[x].v_angle
            full_voltage[iterator + len(self.buses)] = self.buses[x].v_mag
            iterator += 1
        iterator = 0
        slack_position: List[int] = list()
        pv_positions: List[int] = list()
        for c in self.buses:
            if self.buses[c].bus_type == "slack":
                slack_position.extend([iterator])
            elif self.buses[c].bus_type == "PV":
                pv_positions.extend([iterator + len(self.buses)])
            iterator += 1
        size = 2*len(self.buses)-2 - len(pv_positions)
        mismatch = np.zeros(size, dtype=float)
        f_calc = np.zeros(len(self.buses)*2)

        #start calculating real power
        k = 0
        for a in self.buses:
            n = 0
            temp = 0
            for b in self.buses:
                temp += np.abs(self.final_y_bus.loc[a, b]) * np.abs(V[n]) * np.cos(np.angle(V[k]) - np.angle(V[n]) - np.angle(self.final_y_bus.loc[a, b]))
                n += 1
            f_calc[k] = np.abs(V[k]) * temp
            k += 1
        #calc for vars
        k = 0
        for a in self.buses:
            n = 0
            temp = 0
            for b in self.buses:
                temp += np.abs(self.final_y_bus.loc[a, b]) * np.abs(V[n]) * np.sin(np.angle(V[k]) - np.angle(V[n]) - np.angle(self.final_y_bus.loc[a, b]))
                n += 1
            f_calc[k + len(self.buses)] = np.abs(V[k]) * temp
            k += 1


        #make known power vector for comparison
        f = np.zeros(len(self.buses) * 2)
        k = 0
        for x in self.buses:
            for y in self.generators:
                if x == self.generators[y].bus:
                    f[k] = self.generators[y].P
            for z in self.loads:
                if x == self.loads[z].bus:
                    f[k] = -1 * self.loads[z].P
                    f[k + len(self.buses)] = -1 * self.loads[z].Q
            k += 1
        temp_mismatch = f - f_calc
        sp_size = len(slack_position)
        for k in slack_position:
            slack_position.extend([k + len(self.buses)])
            if len(slack_position) >= 2*sp_size:
                break
        pv_size = len(pv_positions)
        del_vector = None



        #delte unwanted power rows
        for k in range(len(f)-1, -1, -1):
            if k in slack_position or k in pv_positions:
                temp_mismatch = np.delete(temp_mismatch, k)
        mismatch = temp_mismatch

        return mismatch

    def calc_solution(self, y, J):
        #solve for delta x
        delt_x = np.linalg.solve(J, y)

        #assign new bus voltages
        iterator = 0
        slack_position: List[int] = list()
        pv_positions: List[int] = list()
        for c in self.buses:
            if self.buses[c].bus_type == "slack":
                slack_position.extend([iterator])
            elif self.buses[c].bus_type == "PV":
                pv_positions.extend([iterator])
            iterator += 1

        i = 0
        for x in self.buses:
            if self.buses[x].bus_type == "slack":
                continue
            else:
                self.buses[x].set_bus_voltage_angle(self.buses[x].v_angle + delt_x[i])
                i += 1

        for x in self.buses:
            if self.buses[x].bus_type == "slack" or self.buses[x].bus_type == "PV":
                continue
            else:
                self.buses[x].set_bus_voltage_mag(self.buses[x].v_mag + delt_x[i])
                i += 1

        a = 0
        new_V = pd.array(data=np.zeros(len(self.buses)), dtype=complex)

        for i in self.buses:
            new_V[a] = self.buses[i].v_mag * np.exp(1j * self.buses[i].v_angle)
            a += 1
        return new_V

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
        skip_k = 0
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

                        if self.buses[p].bus_number == k+1:
                            q += 1
                            continue
                        temp += np.abs(self.final_y_bus.loc[a, p]) * np.abs(V[q]) * np.sin(np.angle(V[k]) - np.angle(V[q]) - np.angle(self.final_y_bus.loc[a, p]))
                        q += 1
                    J1.loc[k, n] = -1 * np.abs(V[k]) * temp
                n += 1

            k += 1
        return J1.to_numpy()

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
        skip_k = 0
        for a in self.buses:
            if self.buses[a].bus_number in slack_position:
                skip_k += 1
                continue
            n = 0
            skip_n = 0
            for b in self.buses:
                if self.buses[b].bus_number in slack_position or self.buses[b].bus_number in pv_positions:
                    skip_n += 1
                    continue
                if k != n:
                    J2.loc[k, n] = np.abs(V[k]) * np.abs(self.final_y_bus.loc[a, b]) * np.cos(np.angle(V[k]) - np.angle(V[n]) - np.angle(self.final_y_bus.loc[a, b]))

                else:
                    temp: float = 0
                    q = 0
                    for p in self.buses:
                        temp += np.abs(self.final_y_bus.loc[a, p]) * np.abs(V[q]) * np.cos(np.angle(V[k]) - np.angle(V[q]) - np.angle(self.final_y_bus.loc[a, p]))
                        q += 1
                    J2.loc[k, n] = np.abs(V[k]) * np.abs(self.final_y_bus.loc[a, a]) * np.cos(np.angle(self.final_y_bus.loc[a, a])) + temp
                n += 1

            k += 1
        return J2.to_numpy()

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

        k = 0

        for a in self.buses:
            if self.buses[a].bus_number in slack_position or self.buses[a].bus_number in pv_positions:
                continue
            n = 0
            for b in self.buses:
                if self.buses[b].bus_number in slack_position:
                    continue
                if k != n:
                    J3.loc[k, n] = -1 * np.abs(V[k]) * np.abs(self.final_y_bus.loc[a, b]) * np.abs(V[n]) * np.cos(np.angle(V[k]) - np.angle(V[n]) - np.angle(self.final_y_bus.loc[a, b]))

                else:
                    temp: float = 0
                    q = 0
                    for p in self.buses:
                        if self.buses[p].bus_number == k+1:
                            q += 1
                            continue
                        temp += np.abs(self.final_y_bus.loc[a, p]) * np.abs(V[q]) * np.cos(np.angle(V[k]) - np.angle(V[q]) - np.angle(self.final_y_bus.loc[a, p]))
                        q += 1
                    J3.loc[k, n] = np.abs(V[k]) * temp
                n += 1

            k += 1

        return J3.to_numpy()

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
        k = 0
        for a in self.buses:
            if self.buses[a].bus_number in slack_position or self.buses[a].bus_number in pv_positions:
                continue
            n = 0
            for b in self.buses:
                if self.buses[b].bus_number in slack_position or self.buses[b].bus_number in pv_positions:
                    continue
                if k != n:
                    J4.loc[k, n] = np.abs(V[k]) * np.abs(self.final_y_bus.loc[a, b]) * np.sin(np.angle(V[k]) - np.angle(V[n]) - np.angle(self.final_y_bus.loc[a, b]))

                else:
                    temp: float = 0
                    q = 0
                    for p in self.buses:
                        temp += np.abs(self.final_y_bus.loc[a, p]) * np.abs(V[q]) * np.sin(np.angle(V[k]) - np.angle(V[q]) - np.angle(self.final_y_bus.loc[a, p]))
                        q += 1
                    J4.loc[k, n] = -1 * np.abs(V[k]) * np.abs(self.final_y_bus.loc[a, a]) * np.sin(np.angle(self.final_y_bus.loc[a, a])) + temp
                n += 1

            k += 1
        return J4.to_numpy()


    def calc_jacobian(self, J1, J2, J3, J4):
        iterator = 0
        slack_position: List[int] = list()
        pv_positions: List[int] = list()
        for c in self.buses:
            if self.buses[c].bus_type == "slack":
                slack_position.extend([iterator])
            elif self.buses[c].bus_type == "PV":
                pv_positions.extend([iterator])
            iterator += 1
        j12 = np.hstack([J1, J2])
        j23 = np.hstack([J3, J4])
        J = np.vstack([j12, j23])
        return J




    def simulate(self):
        self.get_y_bus()
        #initialize bus voltages to 1<0 deg
        V = pd.array(data=np.zeros(len(self.buses)), dtype=complex)
        a = 0
        for i in self.buses:
            if self.buses[i].bus_type != "slack" and self.buses[i].bus_type != "PV":
                self.buses[i].set_bus_voltage_mag(1)
            self.buses[i].set_bus_voltage_angle(0)
            V[a] = self.buses[i].v_mag * np.exp(1j*self.buses[i].v_angle)
            a += 1
        tolerance = 0.0001
        for i in range(500):
            y = self.calc_mismatch(V)
            count = 0
            for k in range(len(y)):
                if np.abs(y[k]) <= tolerance:
                    count += 1
            if count == len(y) or i == 49:
                #create solution vector
                print("Solution found")
                v_solution = pd.array(data=np.zeros(len(self.buses)*2))
                a = 0
                for k in self.buses:
                    v_solution[a] = np.rad2deg(self.buses[k].v_angle)
                    v_solution[a + len(self.buses)] = self.buses[k].v_mag
                    a += 1
                a = 0
                v_solution = pd.DataFrame(data=np.zeros(len(self.buses)), index=self.buses, dtype=complex)
                for k in self.buses:
                    v_solution.loc[k] = self.buses[k].v_mag * np.exp(1j*self.buses[k].v_angle)
                return v_solution
            J1 = self.calc_j1(V)
            J2 = self.calc_j2(V)
            J3 = self.calc_j3(V)
            J4 = self.calc_j4(V)
            J = self.calc_jacobian(J1, J2, J3, J4)
            V = self.calc_solution(y, J)

    def calc_line_currents(self, V):
        for x in self.transmissionLines:
            self.transmissionLines[x].set_line_current(np.abs(V.loc[self.transmissionLines[x].busA.name, 0] * self.transmissionLines[x].y_bus.loc[self.transmissionLines[x].busA.name, self.transmissionLines[x].busA.name] + V.loc[self.transmissionLines[x].busB.name, 0] * self.transmissionLines[x].y_bus.loc[self.transmissionLines[x].busA.name, self.transmissionLines[x].busB.name]))
            #self.transmissionLines[x].set_line_current(np.abs((V.loc[self.transmissionLines[x].busA.name, 0] - V.loc[self.transmissionLines[x].busB.name, 0]) * self.transmissionLines[x].y_bus.loc[self.transmissionLines[x].busA.name, self.transmissionLines[x].busA.name]))
            if self.transmissionLines[x].conductor.current_limit <= self.transmissionLines[x].current * (s.S_mva/(np.sqrt(3) * self.transmissionLines[x].busA.voltage_base)):
                print("WARNING: " + self.transmissionLines[x].name + " EXCEEDS AMPACITY LIMIT")

    def calc_power_flow(self, V):
        size = (len(self.buses), 2)
        d = np.zeros(size)
        i = 0
        bus_names = ['']
        for x in self.buses:
            bus_names.append(self.buses[x].name)
            i += 1
        bus_names.remove('')


        f_calc = pd.DataFrame(data=d, index=bus_names, columns=["P", "Q"], dtype=float)
        # start calculating real power
        k = 0
        for a in self.buses:
            n = 0
            temp = 0
            for b in self.buses:
                temp += np.abs(self.final_y_bus.loc[a, b]) * np.abs(V.loc[self.buses[b].name, 0]) * np.cos(
                    np.angle(V.loc[self.buses[a].name, 0]) - np.angle(V.loc[self.buses[b].name, 0]) - np.angle(self.final_y_bus.loc[a, b]))
                n += 1
            f_calc.loc[a, "P"] = np.abs(V.loc[self.buses[a].name, 0]) * temp
            k += 1
        # calc for vars
        k = 0
        for a in self.buses:
            n = 0
            temp = 0
            for b in self.buses:
                temp += np.abs(self.final_y_bus.loc[a, b]) * np.abs(V.loc[self.buses[b].name, 0]) * np.sin(
                    np.angle(V.loc[self.buses[a].name, 0]) - np.angle(V.loc[self.buses[b].name, 0]) - np.angle(
                        self.final_y_bus.loc[a, b]))
                n += 1
            f_calc.loc[a, "Q"] = np.abs(V.loc[self.buses[a].name, 0]) * temp
            k += 1
        return f_calc

    def calc_line_losses(self):
        d = np.zeros(len(self.transmissionLines))
        tline_names = ['']
        i = 0
        for x in self.transmissionLines:
            tline_names.append(self.transmissionLines[x].name)
            i += 1
        tline_names.remove('')
        losses = pd.DataFrame(data=d, index=tline_names)
        a = 0
        for x in self.transmissionLines:
            losses.loc[x] = self.transmissionLines[x].current ** 2 * np.real(self.transmissionLines[x].get_Z_pu())
            a += 1
        return losses * s.S_mva
