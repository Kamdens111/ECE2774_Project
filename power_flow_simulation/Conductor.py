import Geometry as Geometry
import numpy as np
import sys


class Conductor:

    def __init__(self, GMR, R_c, d_conductor, geometry: Geometry):
        self.GMR = GMR
        self.R_c = R_c
        self.d_conductor = d_conductor
        self.geometry = geometry
        self.Deq = geometry.Deq
        self.num_conductors = geometry.num_conductors
        self.D_sl = 0
        self.D_sc = 0
        self.__calc_params()

    def __calc_params(self):
        r = self.d_conductor/24  # assuming d given in inches
        if self.geometry.num_conductors == 1:
            self.D_sl = self.GMR
            self.D_sc = r
        elif self.geometry.num_conductors == 2:
            self.D_sl = np.sqrt(self.geometry.conductor_distance * self.GMR)
            self.D_sc = np.sqrt(self.geometry.conductor_distance*r)
        elif self.geometry.num_conductors == 3:
            self.D_sl = np.cbrt(self.geometry.conductor_distance**2 * self.GMR)
            self.D_sc = np.cbrt(self.geometry.conductor_distance**2 * r)
        elif self.geometry.num_conductors == 4:
            self.D_sl = 1.091*np.power(self.geometry.conductor_distance**3 * self.GMR, 1/4)
            self.D_sc = 1.091*np.power(self.geometry.conductor_distance**3 * r, 1/4)
        else:
            print("Error: Too many conductors in bundle")
            sys.exit()



