# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:38:47 2019
@author: ericmuckley@gmail.com
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from PyQt5 import uic, QtCore, QtWidgets, QtGui
import pandas as pd
import numpy as np
import sys

fontsize=16
plt.rcParams['xtick.labelsize'] = fontsize 
plt.rcParams['ytick.labelsize'] = fontsize



#path of the .ui Qt designer file 
qt_ui_file = 'C:\\Users\\a6q\\IMES_layout_0.ui'
#timer which updates fields on GUI (set interval in ms)
update_interval = 2000

#load Qt designer XML .ui GUI file
Ui_MainWindow, QtBaseClass = uic.loadUiType(qt_ui_file)


# create the main window
class App(QMainWindow):
    def __init__(self):
        #initialize application
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        #create a timer for updating fields on GUI
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_fields)
        self.timer.start(update_interval)        
        
        #assign actions to top menu items
        self.ui.actionQuit.triggered.connect(self.quit_app)
        
        #assign actions to each button
        self.ui.print_iv_biases.clicked.connect(self.print_iv_biases)
        self.ui.print_cv_biases.clicked.connect(self.print_cv_biases)
        #self.ui.measure_iv.clicked.connect(self.print_table_contents)
        #self.ui.print_table.clicked.connect(self.print_table_contents)
        self.ui.plot_sequence.clicked.connect(Pressure.show_sequence_plot)




    # ------------ system contol functions ------------------------------
        
    def update_fields(self):
        '''update fields peridically as set by timer timeout'''
        #update actual RH field
        self.ui.actual_rh_display.display(self.ui.set_rh.value())
        #update plot of pressure sequence
        #self.update_seq_plot()
    
    
    
    def quit_app(self):
        #quit the application
        self.close()
    
        #app.exec_()
        #sys.exit()
        #sys.exit(app.exec_())


    # --------- functions for electrical characterization -------------------
    
    def calculate_iv_biases(self):
        '''calculate biases to be used for I-V measurements based on
        user-designated initial and final biases and number of bias steps'''
        iv_biases = np.linspace(self.ui.initial_bias.value(),
                        self.ui.final_bias.value(),
                        self.ui.voltage_steps.value())
        return iv_biases
    
    def calculate_cv_biases(self):
        '''calculate biases to be used for C-V measurements based on
        user-designated initial and final biases and number of bias steps'''
        iv_biases = self.calculate_iv_biases()
        cv_biases = np.concatenate((iv_biases, np.flip(iv_biases)[1:],
                                    -iv_biases))
        return cv_biases    
        
    def print_iv_biases(self):
        '''print biases for I-V measurements in the output box'''
        global iv_biases
        iv_biases = self.calculate_iv_biases()
        self.ui.output_box.append('IV biases = '+format(iv_biases))
        
    def print_cv_biases(self):
        '''print biases for C-V measurements in the output box'''
        global cv_biases
        cv_biases = self.calculate_cv_biases()
        self.ui.output_box.append('CV biases = '+format(cv_biases))
    
    
    
    
class Pressure():    
    #------- functions for pressure control ------------------------------
    
    def rh_table_to_df():
        '''convert presure sequence table to pandas dataframe'''
        global rh_df
        #create empty dataframe
        rh_df = pd.DataFrame(columns=['time', 'rh'],
                                index=range(window.pressure_table.rowCount()))
        #populate dataframe
        for rowi in range(window.pressure_table.rowCount()):
            for colj in range(window.pressure_table.columnCount()):
                rh_df.iloc[rowi, colj] = window.pressure_table.item(rowi, colj).text()
        #delete empty rows
        rh_df = rh_df[rh_df['time'] != '0']        
        return rh_df

    def show_sequence_plot():
        '''plot the RH sequence'''
        rh_df = Pressure.rh_table_to_df()
        
        seq_time = np.array(rh_df['time'].astype(float))
        plot_seq_time = np.insert(np.cumsum(seq_time), 0, 0)/60
        
        seq_rh = np.array(rh_df['rh'].astype(float))
        plot_seq_rh = np.insert(seq_rh, 0, seq_rh[0])        

        plt.fill_between(plot_seq_time, plot_seq_rh, step='pre')
        plt.plot(plot_seq_time, plot_seq_rh, c='b', drawstyle='steps', alpha=0)
        plt.xlabel('Time (hours)', fontsize=fontsize)
        plt.ylabel('RH (%)', fontsize=fontsize)
        plt.tight_layout()
        plt.show()




#--------------------------- run application ----------------------------
if __name__ == '__main__':
    
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else: app = QtWidgets.QApplication.instance() 
    
    window = App()
    window.update_fields()
    window.show()
    sys.exit(app.exec_())
