from Transformer import Transformer
from Bus import Bus


bus1 = Bus(0, 125)
bus2 = Bus(1, 345e3)
tx1 = Transformer(400e6, 345e3, 15e3, 8.022, 13.33, bus1, bus2)
tx1.show_params()



