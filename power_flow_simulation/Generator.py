import numpy as np
import Bus as Bus
import pandas as pd
from Settings import s


class Generator:
    def __init__(self, name, bus: str, P: float = s.S_mva):
        self.name = name
        self.bus = bus
        self.P = P/s.S_mva
