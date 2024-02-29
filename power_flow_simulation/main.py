from Transformer import Transformer
from Conductor import Conductor
from Geometry import Geometry
from TransmissionLine import TransmissionLine
from Bus import Bus

bus1 = Bus(0, 500e3)
bus2 = Bus(1, 500e3)
tx1 = Transformer(500e6, 7.5, 13, bus1, bus2)
print("Transformer per unit values:")
tx1.show_params()

bundling1 = Geometry(12.5, 12.5, 20, 1, 3)
Finch = Conductor(0.0435, 0.0969, 1.293, bundling1)
line1 = TransmissionLine(100, Finch, bus1, bus2)
print("Transmission line per unit values:")
line1.show_pu_values()






