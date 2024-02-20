import numpy as np


class Tline:

    def __init__(self, R_prime, X_prime, G_prime, B_prime, length, busA, busB):
        self.R_prime = R_prime
        self.X_prime = X_prime
        self.G_prime = G_prime
        self.B_prime = B_prime
        self.length = length
        self.busA = busA
        self.busB = busB
        self.__calc_params()

    def __calc_params(self):
        self.R=self.R_prime*self.length
        self.X=self.X_prime*self.length
        self.G=self.G_prime*self.length
        self.B=self.B_prime

    def show_params(self):
        print("R = ", self.R, "Ohms")
        print("X = ", self.X, "Ohms")
        print("G = ", self.G, "S")
        print("B = ", self.B, "S")

