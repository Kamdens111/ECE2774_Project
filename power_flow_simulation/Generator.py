import numpy as np
import Bus as Bus
import pandas as pd
from Settings import s


class Generator:
    def __init__(self, name, bus: str,  X1: float = 0.1, P: float = s.S_mva):
        self.name = name
        self.bus = bus
        self.P = P/s.S_mva
        self.X1 = X1
