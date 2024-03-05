import numpy as np
from Settings import s
import Bus as Bus
import pandas as pd


class Transformer:
    # constructor
    def __init__(self, tx_power_rating, percent_z, x_r_ratio, busA: Bus, busB: Bus): #add name
        self.tx_power_rating = tx_power_rating
        self.percent_z = percent_z
        self.x_r_ratio = x_r_ratio
        self.busA = busA
        self.busB = busB
        self.y_bus = np.zeros(2)

        self.buses = [self.busA, self.busB]
        self.__calc_params()

    # private method to calculate the X and R needed for power flow

    def __calc_params(self):
        # find angle
        Z_pu_old = (self.percent_z/100)*np.exp(1j*np.arctan(self.x_r_ratio))
        # Z_sys_base = np.power(self.busB.voltage_base, 2)/s.S_mva  # fix this
        # Z_tx_base = np.power(self.high_v, 2)/self.tx_power_rating
        Z_pu_new = Z_pu_old*(s.S_mva/self.tx_power_rating)
        self.R = np.real(Z_pu_new)
        self.X = np.imag(Z_pu_new)
        Z = self.R + 1j*self.X
        Y = 1/Z

        # calculate y_bus for this system
        prim_y = np.array([[Y, -Y], [-Y, Y]])  # change to pandas
        self.y_bus = pd.DataFrame(prim_y, [self.busA.name, self.busB.name], [self.busA.name, self.busB.name], dtype=complex)

    def get_bus_admittance(self):
        return self.y_bus

    # show calculated r and x
    def show_params(self):
        print("R = ", self.R, "pu")
        print("X = ", self. X, "pu")
