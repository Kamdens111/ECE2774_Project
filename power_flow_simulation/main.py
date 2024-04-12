from PowerFlow import PowerFlow
from Settings import s


sim1 = PowerFlow("Sim")
sim1.add_bus("bus1", 20e3, "slack")
sim1.add_bus("bus2", 230e3)
sim1.add_transformer("T1", 125e6, 8.5, 10, "bus1", "bus2")

#add tline parameters
sim1.add_geometry("geo1", 19.5, 19.5, 39, 1.5, 2)
sim1.add_conductor("Partridge", 0.0217, 0.385, 0.642, 460, "geo1")

sim1.add_bus("bus3", 230e3)
sim1.add_bus("bus4", 230e3)
sim1.add_bus("bus5", 230e3)
sim1.add_bus("bus6", 230e3)
sim1.add_bus("bus7", 18e3, "PV", 1.00)

sim1.add_transmissionLine("tline1", 10, "Partridge", "bus2", "bus4")
sim1.add_transmissionLine("tline2", 25, "Partridge", "bus2", "bus3")
sim1.add_transmissionLine("tline3", 20, "Partridge", "bus3", "bus5")
sim1.add_transmissionLine("tline4", 20, "Partridge", "bus4", "bus6")
sim1.add_transmissionLine("tline5", 10, "Partridge", "bus5", "bus6")
sim1.add_transmissionLine("tline6", 35, "Partridge", "bus4", "bus5")

#add transformer
sim1.add_transformer("T2", 200e6, 10.5, 12, "bus6", "bus7")

sim1.add_generator("G1", "bus1", 0.12)
sim1.add_generator("G2", "bus7", 0.12, 200e6)

sim1.add_load("load1", "bus2", 0, 0)
sim1.add_load("load2", "bus3", 110e6, 50e6)
sim1.add_load("load3", "bus4", 100e6, 70e6)
sim1.add_load("load4", "bus5", 100e6, 65e6)
sim1.add_load("load5", "bus6", 0, 0)

#V = sim1.simulate_powerflow()
#sim1.calc_line_currents(V)
#flow = sim1.calc_power_flow(V) * s.S_mva
#line_losses = sim1.calc_line_losses()
#total_losses = sum(line_losses.to_numpy())
sim1.simulate_fault("bus3")







