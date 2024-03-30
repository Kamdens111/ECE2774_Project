from PowerFlow import PowerFlow


sim1 = PowerFlow("Sim")
sim1.add_bus("bus1", 20e3, "slack")
sim1.add_bus("bus2", 230e3)
sim1.add_transformer("T1", 125e6, 8.5, 10, "bus1", "bus2")

#add tline parameters
sim1.add_geometry("geo1", 19.5, 19.5, 39, 1.5, 2)
sim1.add_conductor("Partridge", 0.0217, 0.385, 0.642, "geo1")

sim1.add_bus("bus3", 230e3)
sim1.add_bus("bus4", 230e3)
sim1.add_bus("bus5", 230e3)
sim1.add_bus("bus6", 230e3)
sim1.add_bus("bus7", 18e3, "PV", 1.00, 2)

sim1.add_transmissionLine("tline1", 10, "Partridge", "bus2", "bus4")
sim1.add_transmissionLine("tline2", 25, "Partridge", "bus2", "bus3")
sim1.add_transmissionLine("tline3", 20, "Partridge", "bus3", "bus5")
sim1.add_transmissionLine("tline4", 20, "Partridge", "bus4", "bus6")
sim1.add_transmissionLine("tline5", 10, "Partridge", "bus5", "bus6")
sim1.add_transmissionLine("tline6", 35, "Partridge", "bus4", "bus5")

#add transformer
sim1.add_transformer("T2", 200e6, 10.5, 12, "bus6", "bus7")

#calc ybus
sim1.get_y_bus()
sim1.simulate()



#bus1 = Bus(0, 500e3)
#bus2 = Bus(1, 500e3)
#tx1 = Transformer(500e6, 7.5, 13, bus1, bus2)
#print("Transformer per unit values:")
#tx1.show_params()

#bundling1 = Geometry(12.5, 12.5, 20, 1, 3)
#Finch = Conductor(0.0435, 0.0969, 1.293, bundling1)
#line1 = TransmissionLine(100, Finch, bus1, bus2)
#print("Transmission line per unit values:")
#line1.show_pu_values()






