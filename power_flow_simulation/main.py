from Transformer import Transformer
from Conductor import Conductor
from Geometry import Geometry
from TransmissionLine import TransmissionLine
from Bus import Bus


bus1 = Bus(0, 138e3)
bus2 = Bus(1, 138e3)
tx1 = Transformer(400e6, 8.022, 13.33, bus1, bus2)

bundling1 = Geometry(12.5, 12.5, 20, 1, 3)
Finch = Conductor(0.0435, 0.0969, 1.293, bundling1)
line1 = TransmissionLine(1, Finch, bus1, bus2)
line1.show_pu_values()
print(line1.get_shunt_admittance())




