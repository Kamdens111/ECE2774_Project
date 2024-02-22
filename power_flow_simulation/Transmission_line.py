import numpy as np
import Bus as Bus
from Settings import s


class Transmission_line:

    def __init__(self, GMR, d, d_ab, d_bc, d_ca, line_length, num_conductors, d_conductor, R_c, busA: Bus, busB: Bus):
        self.d_conductor = d_conductor
        self.R_c = R_c
        self.GMR = GMR
        self.d = d
        self.d_ab = d_ab
        self.d_bc = d_bc
        self.d_ca = d_ca
        self.line_length = line_length
        self.num_conductors = num_conductors
        self.busA = busA
        self.busB = busB
        self.y_bus = np.zeros(2)
        self.__calc_params()

    def __calc_params(self):
        G_prime = 0
        D_eq = np.cbrt(self.d_ab, self.d_bc, self.d_ca)
        # right now only works for 2 conductor bundles
        D_sl = np.power(self.d * self.GMR, 1/self.num_conductors)
        D_sc = np.power(self.d * (self.d_conductor/(2*12)), 1/self.num_conductors)
        R_prime = self.R_c/self.num_conductors
        X_prime = (2 * np.pi * s.frequency) * 2e-7 * np.log(D_eq/D_sl)*1609
        B_prime = (2 * np.pi * s.frequency) * ((2 * np.pi * 8.854e-12)/(np.log(D_eq/D_sc))) * 1609
        self.R = R_prime*self.line_length
        self.X = X_prime*self.line_length
        self.G = G_prime*self.line_length
        self.B = B_prime*self.line_length
        Z = self.R + 1j*self.X
        Y_s = 1/Z
        Y_p = 1/(self.G + 1j*self.B)
        Y_tot = Y_s + Y_p/2

        self.Y_p = np.array([[Y_tot, -1*Y_s], [-1*Y_s, Y_tot]])

    def show_params(self):
        print("R = ", self.R, "Ohms")
        print("X = ", self.X, "Ohms")
        print("G = ", self.G, "S")
        print("B = ", self.B, "S")


