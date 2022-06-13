#%%　imoprt package
import pyvisa as visa
import time  
import matplotlib.pyplot as plt 
import pandas as pd 
import matplotlib.cm as cm
import numpy as np



#%% confirming the address connected to GPIB
rm=visa.ResourceManager()
print(rm.list_resources())


#%% define measurement equipment
AgilentN5183A = rm.open_resource(  'GPIB0::17::INSTR'  ) # signal generator
keithley2182A = rm.open_resource(  'GPIB0::8::INSTR'  ) # nanovoltmeter
ADCMT6240a = rm.open_resource(  'GPIB0::11::INSTR'  ) # source voltage

# initialize
keithley2182A.write("*RST")
AgilentN5183A.write('*RST')
ADCMT6240a.write('C,*RST')

# signal generator
AgilentN5183A.write(':FREQ:MODE CW')
AgilentN5183A.write(':POW:MODE FIX')
AgilentN5183A.write('OUTP ON')

# nanovoltmeter
keithley2182A.write(":SENSe:VOLTage:NPLCycles 1") # medium
keithley2182A.write(":SENSe:VOLTage:DFILter 0") # digital filter off

# source voltage
ADCMT6240a.write('M1')  # trigger mode hold
ADCMT6240a.write('VF')  # voltage output
ADCMT6240a.write('F2')  # current measurement
ADCMT6240a.write('SOV0,LMI0.03')  # dc 0V, limit 30mA
ADCMT6240a.write('OPR') # output on


def single_STFMR(P,f,datapoint, field_slope , field_offset, start=-7,end=7):
    AgilentN5183A.write(':POW %d DBM' %P)
    AgilentN5183A.write(':FREQ %d GHz' %f)
    mag = []
    Vmix =[]
    for i in range(datapoint):
        slope=(end-start)/datapoint
        V=slope*(i-datapoint//2)
        H=field_slope*V+field_offset
        ADCMT6240a.write('SOV%f' % V)
        V_DC=float(keithley2182A.query(":SENSe:Data?")) # read voltage
        mag.append(H)
        Vmix.append(V_DC)
    return mag, Vmix


if __name__ == '__main__':
    single_STFMR(P=20, f=6, datapoint=1400, field_slope=42.7, field_offset=0.1)