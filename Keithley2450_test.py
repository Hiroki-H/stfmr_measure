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

# %% define keithley 2450 current source
keithley2450=rm.open_resource('GPIB0::18::INSTR')
# %% set keithley 2450
current=1e-4 #Ampere
keithley2450.write("*RST")
keithley2450.write(":SOUR:FUNC:MODE CURR")
keithley2450.write(":SOUR:CURR:VLIM 10") #compliance voltage V
keithley2450.write(":SOUR:CURR " + str(current))
keithley2450.write(":OUTP ON")

I=float(keithley2450.query("measure:current?"))