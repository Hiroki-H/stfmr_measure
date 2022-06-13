#%%　imoprt package
import pyvisa as visa
import time
import matplotlib.pyplot as plt
#import pandas as pd
#import matplotlib.cm as cm
import numpy as np

#%% confirming the address connected to GPIB
rm=visa.ResourceManager()
print(rm.list_resources())

#%% define kethley2182A nanovoltmeter
AgilentN5183A = rm.open_resource(  'GPIB0::17::INSTR'  )

AgilentN5183A.write('*RST')

AgilentN5183A.write(':FREQ:MODE CW')

AgilentN5183A.write(':FREQ 1 GHz')

AgilentN5183A.write(':POW:MODE FIX')

AgilentN5183A.write(':POW 0 DBM')

AgilentN5183A.write('OUTP ON')

AgilentN5183A.write('*RST')
