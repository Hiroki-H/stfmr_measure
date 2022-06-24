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
ADCMT6240a = rm.open_resource(  'GPIB0::1::INSTR'  ) # source voltage

# initialize
#keithley2182A.write("*RST")
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

def set_field(field):
    V = float(ADCMT6240a.query('SOV?')[3:])
    dV= abs(field-V)
    i=0
    while dV>0.2:
        #V = float(ADCMT6240a.query('SOV?')[3:])
        if (field-V)>0:
            V +=0.2
        else:
            V -=0.2
        ADCMT6240a.write('SOV%f' % V)
        time.sleep(0.05)
        V = float(ADCMT6240a.query('SOV?')[3:])
        dV= abs(field-V)
        i+=1
    else:
        ADCMT6240a.write('SOV%f' % field)

def single_STFMR(P,f,datapoint, field_slope , field_offset, start=-6,end=6):
    AgilentN5183A.write(':POW %d DBM' %P)
    AgilentN5183A.write(':FREQ %d GHz' %f)
    set_field(start)
    time.sleep(0.5)
    mag = []
    Vmix =[]
    for i in range(datapoint+1):
        slope=(end-start)/datapoint
        V=slope*(i-datapoint//2)
        H=field_slope*V+field_offset
        ADCMT6240a.write('SOV%f' % V)
        time.sleep(0.05)
        V_DC=float(keithley2182A.query(":SENSe:Data?")) # read voltage
        mag.append(H)
        Vmix.append(V_DC)
        update(H,V_DC)
    return mag, Vmix

x=[]
y=[]
def update(x1,y1):
    global curve, ptr, x ,y, V
    #   Xm[:-1] = Xm[1:]                      # shift data in the temporal mean 1 sample left
    #keithley2450.write(":SOUR:VOLT " + str(V))
    #yvalue = keithley2450.query(":READ? \"defbuffer1\" ,source")
    #xvalue =  keithley2450.query(":READ?")       # read line (single value) from the serial port
    x.append(float(x1))
    y.append(float(y1))
    #   Xm[-1] = float(value)                 # vector containing the instantaneous values
    #   ptr += 1                              # update x position for displaying the curve
    curve.setData(x,y)                     # set the curve with this data
    #curve.setPos(0,float(x[-1]))                   # set x position in the graph to 0
    pg.QtGui.QApplication.processEvents()    # you MUST process the plot now




if __name__ == '__main__':
    ### START QtApp #####
    app = pg.QtGui.QApplication([])            # you MUST do this once (initialize things)
    ####################
    win = pg.GraphicsWindow(title='Signal from serial port') # creates a window
    p = win.addPlot(title='Realtime plot')  # creates empty space for the plot in the window
    curve = p.plot(pen=None,symbol='o')
    for i in range(4,5,1):
        H,V = single_STFMR(P=20, f=i, datapoint=1200, field_slope=42.8, field_offset=-0.1)
        V = np.array(V)
        plt.plot(H,V*1e+6,'-o',label= str(i)+'GHz')
        plt.xlabel('μ0H(mT)',fontsize=20)
        plt.ylabel('V_DC(uV)',fontsize=20)
        plt.legend()
        plt.tight_layout()
    set_field(0)
    AgilentN5183A.write('OUTP OFF')
    ADCMT6240a.write('SBY')
    AgilentN5183A.close()
    keithley2182A.close()
    ADCMT6240a.close()
    plt.show()
    #set_field(7)
