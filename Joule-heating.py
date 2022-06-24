#%%　imoprt package
import pyvisa as visa
import time
import matplotlib.pyplot as plt
#import pandas as pd
import matplotlib.cm as cm
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

#%% confirming the address connected to GPIB
rm=visa.ResourceManager()
print(rm.list_resources())


#%% define measurement equipment
AgilentN5183A = rm.open_resource(  'GPIB0::19::INSTR'  ) # signal generator
keithley2182A = rm.open_resource(  'GPIB0::6::INSTR'  ) # nanovoltmeter
keithley2450 = rm.open_resource(  'GPIB0::1::INSTR'  ) # source voltage


# nanovoltmeter
keithley2182A.write(":SENSe:VOLTage:NPLCycles 1") # medium
keithley2182A.write(":SENSe:VOLTage:DFILter 0") # digital filter off

# %% set keithley 2450
current=1e-4 #Ampere
keithley2450.write("*RST")
keithley2450.write(":SOUR:FUNC:MODE CURR")
keithley2450.write(":SOUR:CURR:VLIM 50") #compliance voltage V
keithley2450.write(":SOUR:CURR " + str(current))
keithley2450.write(":OUTP ON")

I=float(keithley2450.query("measure:current?"))



def JH_DC(amplitude , dI, waittime):
    minimum =0.4 #mA
    keithley2450.write(":OUTP ON")
    for current in np.arange(minimum, amplitude, dI):
        current*=1e-3
        keithley2450.write(":SOUR:CURR %d" %current)
        time.sleep(waittime)
        I_p=float(keithley2450.query("measure:current?"))
        V_p=float(keithley2182A.query(":SENSe:Data?")) # read voltage
        keithley2450.write(":SOUR:CURR %d" %-current)
        time.sleep(waittime)
        I_m=float(keithley2450.query("measure:current?"))
        V_m=float(keithley2182A.query(":SENSe:Data?")) # read voltage

