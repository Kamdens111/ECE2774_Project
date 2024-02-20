
#fix this later
class Settings:
    def __init__(self):
        self._S_mva=100e6
        self._f=60

    @property
    def S_mva(self):
        return self._S_mva

    @property
    def frequency(self):
        return self._f


s=Settings()
