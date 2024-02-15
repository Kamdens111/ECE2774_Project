import numpy as np
import Bus as Bus
import pandas as pd


class Tx:
    # constructor
    def __init__(self, tx_power_rating, high_v, low_v, percent_z, x_r_ratio, busA, busB):
        self.tx_power_rating = tx_power_rating
        self.high_v = high_v
        self.low_v = low_v
        self.percent_z = percent_z
        self.x_r_ratio = x_r_ratio
        self.busA = busA
        self.busB = busB
        self.__calc_params()

    # private method to calculate the X and R needed for power flow

    def __calc_params(self):
        # find angle
        Z_pu_old = (self.percent_z/100)*np.exp(1j*np.arctan(self.x_r_ratio))
        Z_sys_base = np.power(self.high_v, 2)/100e6  # fix this
        Z_tx_base = np.power(self.high_v, 2)/self.tx_power_rating
        Z_pu_new = Z_pu_old*(Z_tx_base/Z_sys_base)
        self.R = np.real(Z_pu_new)
        self.X = np.imag(Z_pu_new)

    # show calculated r and x
    def show_params(self):
        print("R = ", self.R)
        print("X = ", self. X)
