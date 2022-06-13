#%%　imoprt package
import pyvisa as visa
import time  
import matplotlib.pyplot as plt 
import pandas as pd 
import matplotlib.cm as cm
import numpy as np


rm=visa.ResourceManager() 
print(rm.list_resources())

# %% define bipolar power supply
adc_OOP = rm.open_resource(  'GPIB0::11::INSTR'  )

# %% set biplor power supply
adc_OOP.write('C,*RST')
adc_OOP.write('M1')  # trigger mode hold
adc_OOP.write('VF')  # voltage output
adc_OOP.write('F2')  # current measurement
adc_OOP.write('SOV0,LMI0.03')  # dc 0V, limit 30mA
adc_OOP.write('OPR') # output on
def AMR_field(start,end,datapoint):
    data=[]
    field=[]
    adc_OOP.write('SOV%f' % start)
    time.sleep(10)
    for i in range(datapoint):
        slope=(end-start)/datapoint
        V=slope*(i-datapoint//2)
        adc_OOP.write('SOV%f' % V)
        #print(kikusui_IP.query("VOLT?"))
        field.append(V)
    adc_OOP.write('SOV%f' % 0)
    return field, data