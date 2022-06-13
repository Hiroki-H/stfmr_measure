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

#%% define kethley2182A nanovoltmeter
keithley2182A = rm.open_resource(  'GPIB0::8::INSTR'  )
#%% set keithley2182
#keithley2182A.write("*RST")
keithley2182A.write(":SENSe:VOLTage:NPLCycles 1") # medium
keithley2182A.write(":SENSe:VOLTage:DFILter 0") # digital filter off

Volt=float(keithley2182A.query(":SENSe:Data?")) # read voltage