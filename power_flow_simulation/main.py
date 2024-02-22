from Transformer import Transformer
from Bus import Bus


bus1 = Bus(0, 125)
bus2 = Bus(1, 345e3)
tx1 = Transformer(400e6, 8.022, 13.33, bus1, bus2)
print(bus1.bus_number)
bus2.show_bus_number()



