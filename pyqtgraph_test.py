import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

# #　(1) PyQtのウインドウ生成
# app = QtGui.QApplication([])
# #　(2) GraphicsLayoutWidged オブジェクト生成
# win = pg.GraphicsLayoutWidget()
# win.show()
# # (3) PlotItem オブジェクト生成
# plot = win.addPlot(title="real-time scatter plot")
# # (4) PlotDataItemオブジェクト生成
# curve = plot.plot(symbol ='o')
#
# # 初期データ（numpy.ndarrayオブジェクト）生成
# n_data = 1000
# x = 0
# y = 0
#
# # (5) 更新処理を記述した関数
# def update():
#     global x, y, curve, plot
#
#     # x, y を更新
#     x += 1
#     y += 10
#     # データ更新
#     curve.setData(x, y)
#
# # (6) 定期的に(5)の関数を実行
# fps = 60
# timer = QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(1/fps * 1000)
#
#
# # アプリケーション実行
# if __name__ == '__main__':
#     import sys
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()


### START QtApp #####
app = pg.QtGui.QApplication([])            # you MUST do this once (initialize things)
####################
win = pg.GraphicsWindow(title='Signal from serial port') # creates a window
p = win.addPlot(title='Realtime plot')  # creates empty space for the plot in the window
curve = p.plot(pen=None,symbol='o')

#pg.QtGui.QApplication.exec_()
             # create an empty “plot” (a curve to plot)
#windowWidth = 100                       # width of the window displaying the curve
#Xm = np.linspace(0,0,windowWidth)           # create array that will contain the relevant time series
#ptr = windowWidth
V=0
x=[]
y=[]
def update():
  global curve, ptr, x ,y, V
#   Xm[:-1] = Xm[1:]                      # shift data in the temporal mean 1 sample left
  V+=1
  #keithley2450.write(":SOUR:VOLT " + str(V))
  #yvalue = keithley2450.query(":READ? \"defbuffer1\" ,source")
  #xvalue =  keithley2450.query(":READ?")       # read line (single value) from the serial port
  x.append(float(V))
  y.append(float(np.random.rand()))
#   Xm[-1] = float(value)                 # vector containing the instantaneous values
#   ptr += 1                              # update x position for displaying the curve
  curve.setData(x,y)                     # set the curve with this data
  #curve.setPos(0,float(x[-1]))                   # set x position in the graph to 0
  pg.QtGui.QApplication.processEvents()    # you MUST process the plot now
#pg.QtGui.QApplication.exec_() # you MUST put this at the end
        # set first x position
# Realtime data plot. Each time this function is called, the data display is updated
#def update():
#   global curve, ptr, Xm
#   Xm[:-1] = Xm[1:]                      # shift data in the temporal mean 1 sample left
#   value =  keithley2450.ask(":READ?")       # read line (single value) from the serial port
#   Xm[-1] = float(value)                 # vector containing the instantaneous values
#   ptr += 1                              # update x position for displaying the curve
#   curve.setData(Xm)                     # set the curve with this data
#   curve.setPos(0,ptr)                   # set x position in the graph to 0
#   QtGui.QApplication.processEvents()    # you MUST process the plot now
### MAIN PROGRAM #####
# this is a brutal infinite loop calling your realtime data plot
X=0
while X<1000:
   update()
   X+=1
### END QtApp ####
pg.QtGui.QApplication.exec_() # you MUST put this at the end
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    pg.QtGui.QApplication.instance().exec_()

# if __name__ == '__main__':
#     pg.exec()
