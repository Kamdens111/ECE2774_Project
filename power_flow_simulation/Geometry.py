import numpy as np


class Geometry:
    def __init__(self, d_ab, d_bc, d_ca, conductor_distance, num_conductors):
        self.d_ab = d_ab
        self.d_bc = d_bc
        self.d_ca = d_ca
        self.conductor_distance = conductor_distance
        self.num_conductors = num_conductors
        self.Deq = 0
        self.__calc_params()

    def __calc_params(self):
        self.Deq = np.cbrt(self.d_ab * self.d_bc * self.d_ca)


