#Simple Circuit

import numpy as nd
import numpy as np

#create inputs
V_s=100
R_line=5
P_load=2000
V_load=100
R_load=np.power(V_load,2)/P_load

#solve for outputs
I=V_s/(R_line+R_load)
Vout=V_s*R_load/(R_line+R_load)
#print('Vout=',Vout, 'V')
#print('I=',I,'A')

class Bus:
    def __init__(self, number):
        self.number= number

class Line:
    def __init__(self,num, Z):
        self.Z=Z
        self.num=num
    def add(self, impedance):
        self.Z=self.Z+impedance
        return print("New impedance is", self.Z)
    def outputs(self):
        out_V=V_s*R_load/(self.Z+R_load)
        out_I=out_V/R_load
        return print("Vout=", out_V,"V",  " I=", out_I, "A")

a=Bus(0)
b=Bus(1)
c=Line(1,R_line)
c.outputs()
c.add(R_line)
c.outputs()
c.add(10)
c.outputs()


