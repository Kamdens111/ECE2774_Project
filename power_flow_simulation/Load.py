import numpy as np
import Bus as Bus
import pandas as pd
from Settings import s

class Load:
    def __init__(self, name, bus: str, P, Q):
        self.name = name
        self.P = P/s.S_mva
        self.Q = Q/s.S_mva
        self.bus = bus
