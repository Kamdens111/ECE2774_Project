import numpy as np
import Bus as Bus
from Settings import s
import Conductor as Conductor


class TransmissionLine:

    def __init__(self, line_length, conductor: Conductor, busA: Bus, busB: Bus):
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
        Z = self.R + 1j*self.X
        Y_s = 1/Z
        Y_p = 1/(self.G + 1j*self.B)
        Y_tot = Y_s + Y_p/2

        self.y_bus = np.array([[Y_tot, -1*Y_s], [-1*Y_s, Y_tot]])

    def show_params(self):
        print("R = ", self.R, "Ohms")
        print("X = ", self.X, "Ohms")
        print("G = ", self.G, "S")
        print("B = ", self.B, "S")


