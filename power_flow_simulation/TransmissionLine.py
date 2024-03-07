import numpy as np
import Bus as Bus
from Settings import s
import Conductor as Conductor
import pandas as pd


class TransmissionLine:

    def __init__(self, line_length, conductor: Conductor, busA: Bus, busB: Bus): #add name
        self.line_length = line_length
        self.conductor = conductor
        self.busA = busA
        self.busB = busB
        self.y_bus = np.zeros(2)
        self.buses = [self.busA, self.busB]
        self.__calc_params()

    def __calc_params(self):
        G_prime = 0
        R_prime = self.conductor.R_c/self.conductor.num_conductors
        X_prime = (2 * np.pi * s.frequency) * 2e-7 * np.log(self.conductor.Deq/self.conductor.D_sl)*1609
        B_prime = (2 * np.pi * s.frequency) * ((2 * np.pi * 8.854e-12)/(np.log(self.conductor.Deq/self.conductor.D_sc))) * 1609
        self.R = R_prime*self.line_length
        self.X = X_prime*self.line_length
        self.G = G_prime*self.line_length
        self.B = B_prime*self.line_length
        zbase = self.busA.voltage_base ** 2 / s.S_mva
        ybase = 1/zbase

        Rpu = self.R/zbase
        Xpu = self.X/zbase
        Bpu = self.B/ybase

        Y_s = self.get_series_admittance()
        Y_p = self.get_shunt_admittance()
        Y_tot = Y_s + Y_p / 2
        prim_y = np.array([[Y_tot, -1 * Y_s], [-1 * Y_s, Y_tot]])
        self.y_bus = pd.DataFrame(prim_y, [self.busA.name, self.busB.name], [self.busA.name, self.busB.name],
                                  dtype=complex)

    def get_Z_pu(self):
        return (self.R + 1j*self.X)/(self.busA.voltage_base**2 / s.S_mva)

    def get_Y_pu(self):
        return (self.G + 1j*self.B)/(s.S_mva/self.busA.voltage_base**2)

    def get_series_admittance(self):
        Z = self.R + 1j*self.X

        Z_base = self.busA.voltage_base ** 2 / s.S_mva
        return 1/(Z/Z_base)  #series admittance

    def get_shunt_admittance(self):
        Y = self.G + 1j * self.B
        Z_base = self.busA.voltage_base ** 2 / s.S_mva
        Y_base = 1 / Z_base
        return Y / Y_base  # parallel/shunt admittance

    def get_bus_admittance(self):
        return self.y_bus

    def show_pu_values(self):
        print("R = ", np.real(self.get_Z_pu()), "pu")
        print("X = ", np.imag(self.get_Z_pu()), "pu")
        print("G = ", np.real(self.get_Y_pu()), "pu")
        print("B = ", np.imag(self.get_Y_pu()), "pu")


