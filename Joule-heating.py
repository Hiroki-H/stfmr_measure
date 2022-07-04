#%%　imoprt package
from multiprocessing.connection import wait
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
keithley2450 = rm.open_resource(  'GPIB0::18::INSTR'  ) # source voltage


# nanovoltmeter
keithley2182A.write(":SENSe:VOLTage:NPLCycles 1") # medium
keithley2182A.write(":SENSe:VOLTage:DFILter 0") # digital filter off

# %% set keithley 2450
keithley2450.write("*RST")
keithley2450.write(":SOUR:FUNC:MODE CURR")
keithley2450.write(":SOUR:CURR:VLIM 20") #compliance voltage V


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


def JH_DC(minimum, maximum , dI, waittime):
    keithley2450.write(":OUTP ON")
    I=[]
    R=[]
    for current in np.arange(minimum, maximum, dI):
        current*=1e-3
        keithley2450.write(":SOUR:CURR %f" %current)
        time.sleep(waittime)
        I_p=float(keithley2450.query("measure:current?"))
        V_p=float(keithley2182A.query(":SENSe:Data?")) # read voltage
        I.append(I_p)
        R.append(V_p/I_p)
        update(I_p, V_p/I_p)
        keithley2450.write(":SOUR:CURR %f" %-current)
        time.sleep(waittime)
        I_m=float(keithley2450.query("measure:current?"))
        V_m=float(keithley2182A.query(":SENSe:Data?")) # read voltage
        I.append(I_m)
        R.append(V_m/I_m)
        update(I_m, V_m/I_m)
    keithley2450.write(":OUTP ON")
    keithley2450.write("*RST")
    keithley2450.close()
    keithley2182A.close()
    return I,R

def mW_to_dBm(P_mW):
    P_dBm=10*np.log10(P_mW)
    return P_dBm

def dBm_to_mW(dBm):
    P_mW=10**(dBm/10)
    return P_mW

def JH_RF(small_DC,frequency,minimum_P, maximum_P , dP, waittime):
    keithley2450.write(":OUTP ON")
    I_s=small_DC*1e-3
    keithley2450.write(":SOUR:CURR %f" %I_s)
    AgilentN5183A.write(':FREQ:MODE CW')
    AgilentN5183A.write(':POW:MODE FIX')
    AgilentN5183A.write('OUTP ON')
    AgilentN5183A.write(':FREQ %f GHz' %frequency)
    P=[]
    R=[]
    for RF_P in np.arange(minimum_P, maximum_P, dP):
        P_dBm= mW_to_dBm(RF_P)
        AgilentN5183A.write(':POW %f DBM' %P_dBm)
        time.sleep(waittime)
        PmW= dBm_to_mW(float(AgilentN5183A.query(':POW?')))
        I_dc=float(keithley2450.query("measure:current?"))
        V_dc=float(keithley2182A.query(":SENSe:Data?")) # read voltage
        P.append(PmW)
        R.append(V_dc/I_dc)
        update(PmW, V_dc/I_dc)
    keithley2450.write(":OUTP OFF")
    AgilentN5183A.write('OUTP OFF')
    keithley2450.write("*RST")
    keithley2450.close()
    keithley2182A.close()
    return P,R

def JH_DC_execute(minimum, maximum , dI, waittime):
    global curve, ptr, x ,y, V
    ### START QtApp #####
    app = pg.QtGui.QApplication([])            # you MUST do this once (initialize things)
    ####################
    win = pg.GraphicsWindow(title='Signal from serial port') # creates a window
    p = win.addPlot(title='Realtime plot')  # creates empty space for the plot in the window
    curve = p.plot(pen=None,symbol='o')
    I,R = JH_DC(minimum=minimum, maximum=maximum,dI=dI,waittime=waittime)
    plt.plot(I,R,'o')
    plt.xlabel('I(mA)',fontsize=20)
    plt.ylabel('R(ohm)',fontsize=20)
    plt.tight_layout()
    plt.show()

def JH_RF_exceute(small_DC,frequency,minimum_P, maximum_P , dP, waittime):
    global curve, ptr, x ,y, V
    ### START QtApp #####
    app = pg.QtGui.QApplication([])            # you MUST do this once (initialize things)
    ####################
    win = pg.GraphicsWindow(title='Signal from serial port') # creates a window
    p = win.addPlot(title='Realtime plot')  # creates empty space for the plot in the window
    curve = p.plot(pen=None,symbol='o')
    I,R = JH_RF(small_DC=small_DC,frequency=frequency,minimum_P=minimum_P, maximum_P=maximum_P, dP=dP,waittime=waittime)
    plt.plot(I,R,'o')
    plt.xlabel('I(mA)',fontsize=20)
    plt.ylabel('R(ohm)',fontsize=20)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #JH_DC_execute(minimum=0.4,maximum=20,dI=0.5,waittime=5)
    #print(AgilentN5183A.query(':FREQ?'))
    #print(AgilentN5183A.query(':POW?'))
    JH_RF_exceute(small_DC=0.4,frequency=0.5,minimum_P=10, maximum_P=200, dP=10,waittime=5)