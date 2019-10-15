# -*- coding: utf-8 -*-
"""
Created on Wed Jan 9 14:38:47 2019
@author: ericmuckley@gmail.com
"""

#from alicat import FlowController

from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
fontsize=16
plt.rcParams['xtick.labelsize'] = fontsize 
plt.rcParams['ytick.labelsize'] = fontsize



#path of the .ui Qt designer file 
qt_ui_file = 'C:\\Users\\a6q\\IMES_layout_0.ui'

#load Qt designer XML .ui GUI file
Ui_MainWindow, QtBaseClass = uic.loadUiType(qt_ui_file)

#set up master dataframe to hold all pressure data
master_headers = ['date', 'time', 'pressure', 'temp', 'bias', 'current',
                  'save', 'note']
master_df = pd.DataFrame(columns=master_headers,
            data=np.zeros((10000,len(master_headers))).astype(str))


main_i = 0 #initialize main loop counter
app_start_time = time.time() #initialize main loop timer


class A(object):
    
    butt = 5
    
    def print_hello(self):
       window.ui.output_box.append('hellllo')
    


# create the main window
class App(QMainWindow):

    def __init__(self):
        #initialize application
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        

        
        #create timer which updates fields on GUI (set interval in ms)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.main_loop)
        self.main_loop_delay = self.ui.set_main_loop_delay.value()#1000
        self.timer.start(self.main_loop_delay)        
        
        #assign functions to top menu items
        #example: self.ui.MENU_ITEM_NAME.triggered.connect(self.FUNCTION_NAME)
        self.ui.file_quit.triggered.connect(self.quit_app)
        self.ui.file_set_save.triggered.connect(self.set_file_save_directory)
        self.ui.seq_plot.triggered.connect(self.show_sequence_plot)
        self.ui.seq_export.triggered.connect(self.export_sequence)
        self.ui.seq_import.triggered.connect(self.import_sequence)
        self.ui.seq_clear.triggered.connect(self.clear_sequence)
        self.ui.seq_run.triggered.connect(self.run_sequence)
        self.ui.seq_stop.triggered.connect(self.stop_sequence)
        self.ui.show_pressure_plot.triggered.connect(self.plot_pressure)
        self.ui.preview_iv_biases.triggered.connect(self.print_iv_biases)
        self.ui.preview_cv_biases.triggered.connect(self.print_cv_biases)
        self.ui.measure_iv_button.triggered.connect(self.measure_iv)
        self.ui.measure_cv_button.triggered.connect(self.measure_cv)
        self.ui.measure_current_now.triggered.connect(self.begin_current_measurement)
        self.ui.measure_current_now.triggered.connect(self.measure_current)
        self.ui.view_file_save_dir.triggered.connect(self.print_file_save_dir)
        

        
        #assign actions to GUI buttons
        #example: self.ui.BUTTON_NAME.clicked.connect(self.FUNCTION_NAME)
        self.ui.print_hi.clicked.connect(self.get_harmonics)
        self.ui.print_hi.clicked.connect(A.print_hello)

        #initialize some settings
        self.ui.app_start_time_display.setText(time.ctime())
        self.ui.seq_stop.setDisabled(True)
        self.ui.seq_run.setDisabled(False)
        self.stop_sequence = False



        


    # ------------ system control functions ------------------------------
    def main_loop(self):
        # Main loop to execute which keeps the app running.
        global main_i
        global master_df

        #update indicator fields on GUI
        self.update_fields()


        #measure current 
        if self.ui.measure_current_now.isChecked():
            self.measure_current()       
            
            
            
        #save data
        if self.ui.save_data_now.isChecked():
            try: save_file_dir #check if file save directory has been set yet
            except: self.set_file_save_directory()
            master_df['save'].iloc[main_i] = 'on' #mark current row as "saved"
            #cut off extra dataframe rows which were not filled and export df
            save_master_df = master_df[master_df['save'] == 'on']    
            save_master_df.to_pickle(save_file_dir+'/master_df.pkl')
            self.ui.rows_of_saved_data_display.setText(
                    str(len(save_master_df)))
        
        
        main_i += 1


    
    
    
    
    def update_fields(self):
        #update fields during each loop iteration
        
        #update total sequence time
        pressure_df = self.pressure_table_to_df()
        tot_seq_hrs = pressure_df['time'].astype(float).sum()/60
        self.ui.tot_seq_time_display.setText(
                str(np.round(tot_seq_hrs, decimals=2)))
        self.ui.main_loop_counter_display.setText(str(main_i))
        
        #update actual RH field
        self.ui.actual_rh_display.setText(
                str(np.round(self.ui.set_rh.value(), decimals=3)))
        #display(self.ui.set_rh.value())
        
        #switching between pressure and valve mode
        if self.ui.pressure_mode.isChecked():
            self.ui.set_pressure.setDisabled(False)
            self.ui.set_valve_position.setDisabled(True)
        if self.ui.valve_mode.isChecked():
            self.ui.set_valve_position.setDisabled(False)
            self.ui.set_pressure.setDisabled(True)
            
        #control pressure
        if self.ui.pressure_control_on.isChecked():
            pressure0 = np.random.random()+760
            master_df['pressure'].iloc[main_i] = str(pressure0)
            
            #control pop-up pressure plot
            if not plt.fignum_exists(2):
                    self.ui.show_pressure_plot.setChecked(False)
            if self.ui.show_pressure_plot.isChecked():
                if main_i % 3 == 0:
                    self.plot_pressure()  
            
            
        #timer which updates fields on GUI (set interval in ms)
        self.main_loop_delay = self.ui.set_main_loop_delay.value()
    
        #record date/time and elapsed time at each iteration
        master_df['date'].iloc[main_i] = time.ctime()
        master_df['time'].iloc[main_i] = str(
                np.round(time.time() - app_start_time, decimals=3))
    
    
    
    

    def plot_results(self, x=[1,2,3], y=[2,6,9],
                     xtitle='X', ytitle='Y', plottitle='Title'):
        #plot results of a measurement
        plt.cla()
        plt.ion()
        plt.plot(x, y, c='k', lw=1, marker='o', markersize=5)
        plt.xlabel(xtitle, fontsize=fontsize)
        plt.ylabel(ytitle, fontsize=fontsize)
        plt.title(plottitle, fontsize=fontsize)
        plt.tight_layout()
        plt.draw()
        
    def set_file_save_directory(self):
        #set the directory for saving data files
        global save_file_dir
        save_file_dir = str(QFileDialog.getExistingDirectory(
                            self, 'Select directory for saving files'))
        self.ui.output_box.append('Save file directory set to:')
        self.ui.output_box.append(save_file_dir)

    def print_file_save_dir(self):
        #print the file saving directory
        try: #if save file directory is already set
            self.ui.output_box.append('Current save file directory:')
            self.ui.output_box.append(save_file_dir)
        except: #if file directory is not set
            self.ui.output_box.append(
                    'No save file directory has been set.')
            self.ui.output_box.append(
                    'Please set in File --> Set file save directory.')

    def quit_app(self):
        global master_df
        #quit the application
        self.ui.measure_current_now.setChecked(False) #stop measuring current
        plt.close('all') #close all figures
        self.close() #close app window






    # --------- functions for electrical characterization -------------------
    
    def begin_current_measurement(self):
        #begin current measurement using Keithley-2420 multimeter
        if self.ui.keithley2420_on.isChecked():
            if self.ui.measure_current_now.isChecked():
                #disable other electrical measurements
                self.ui.measure_iv_button.setDisabled(True)
                self.ui.measure_cv_button.setDisabled(True)
                self.ui.output_box.append(
                        'Current measurement at constant bias started.')
            if not self.ui.measure_current_now.isChecked():
                #enable other electrical measurements
                self.ui.measure_iv_button.setDisabled(False)
                self.ui.measure_cv_button.setDisabled(False)
                self.ui.output_box.append(
                        'Current measurement at constant bias ended.')
        else: 
            self.ui.output_box.append('Keithley not connected.')
            
    def measure_current(self):
        #measure current continuously using Keithley-2420 multimeter
        #bias = self.ui.set_constant_bias.value()
        if self.ui.measure_current_now.isChecked():
            current0 = np.random.random()
    
            self.ui.current_display.setText(str(np.round(current0, decimals=9)))
            master_df['current'].iloc[main_i] = current0
    
            #plot current over time
            plt.ion
            fig_current = plt.figure(3)
            fig_current.clf()
            df_current = master_df[master_df['current'] != '0.0']
            
            plt.plot(df_current['time'].astype(float),
                     df_current['current'].astype(float),
                     c='r', lw=1)
            plt.xlabel('Elapsed time (min)', fontsize=fontsize)
            plt.ylabel('Current (A)', fontsize=fontsize)
            fig_current.canvas.set_window_title('Sample current')
            plt.tight_layout()
            plt.draw()
    
    
    def calculate_iv_biases(self):
        #calculate biases to be used for I-V measurements based on
        #user-designated initial and final biases and number of bias steps
        iv_biases_low = -np.linspace(self.ui.max_bias.value(), 0,
                        self.ui.voltage_steps.value())
        iv_biases_high = np.linspace(0, self.ui.max_bias.value(),
                        self.ui.voltage_steps.value())
        iv_biases = np.concatenate((iv_biases_low[:-1], iv_biases_high))
        return iv_biases
    
    def print_iv_biases(self):
        #print biases for I-V measurements in the output box
        iv_biases = self.calculate_iv_biases()
        self.ui.output_box.append('IV biases = '+format(iv_biases))
        plt.ion
        fig_iv_b = plt.figure(3)
        fig_iv_b.clf()
        plt.plot(np.arange(len(iv_biases))+1, iv_biases,
                 c='k', lw=1, marker='o', markersize=5)
        plt.xlabel('Point number', fontsize=fontsize)
        plt.ylabel('Bias (V)', fontsize=fontsize)
        fig_iv_b.canvas.set_window_title('Biases for I-V measurements')
        plt.tight_layout()
        plt.draw()


    def calculate_cv_biases(self):
        #calculate biases to be used for C-V measurements based on
        #user-designated initial and final biases and number of bias steps
        cv_bias_segment = np.linspace(0, self.ui.max_bias.value(),
                        self.ui.voltage_steps.value())
        cv_biases = np.concatenate((cv_bias_segment,
                                    np.flip(cv_bias_segment,axis=0)[1:],
                                    -cv_bias_segment[1:], 
                                    -np.flip(cv_bias_segment,axis=0)[1:],
                                    cv_bias_segment[1:],
                                    np.flip(cv_bias_segment,axis=0)[1:]))
        return cv_biases
    
    def print_cv_biases(self):
        #print biases for C-V measurements in the output box
        cv_biases = self.calculate_cv_biases()
        self.ui.output_box.append('CV biases = '+format(cv_biases))
        plt.ion
        fig_cv_b = plt.figure(4)
        fig_cv_b.clf()
        plt.plot(np.arange(len(cv_biases))+1, cv_biases,
                 c='k', lw=1, marker='o', markersize=5)
        plt.xlabel('Point number', fontsize=fontsize)
        plt.ylabel('Bias (V)', fontsize=fontsize)
        fig_cv_b.canvas.set_window_title('Biases for C-V measurements')
        plt.tight_layout()
        plt.draw()
    
    
    
    
    def measure_iv(self):
        #measure I-V curve
        iv_biases = self.calculate_iv_biases()
        #create array to hold IV data
        global iv_results0
        iv_results0 = np.empty((len(iv_biases), 2))
        iv_results0[:,0] = iv_biases
        self.ui.output_box.append('Measuring I-V...')
        #run the measurement by sweeping through each bias
        for ii, bias0 in enumerate(iv_biases):
            current0 = bias0+np.random.random()/100
            self.ui.current_display.setText(str(np.round(current0, decimals=9)))
            iv_results0[ii, 1] = current0
        
        #plot results
        self.plot_results(x=iv_results0[:,0], y=iv_results0[:,1],
                          xtitle='Bias (V)', ytitle='Current (A)',
                          plottitle='I-V measurement')
        self.ui.output_box.append('I-V measurement complete.')
        
        
        
        
        
    def measure_cv(self):
        #measure C-V curve
        cv_biases = self.calculate_cv_biases()
        #create array to hold IV data
        global cv_results0
        cv_results0 = np.empty((len(cv_biases), 2))
        cv_results0[:,0] = cv_biases
        self.ui.output_box.append('Measuring C-V...')
        #run the measurement by sweeping through each bias
        for ii, bias0 in enumerate(cv_biases):
            current0 = bias0+np.random.random()/100
            self.ui.current_display.setText(str(np.round(current0, decimals=9)))
            cv_results0[ii, 1] = current0
        #plot results
        self.plot_results(x=cv_results0[:,0], y=cv_results0[:,1],
                          xtitle='Bias (V)', ytitle='Current (A)',
                          plottitle='C-V measurement')
        self.ui.output_box.append('C-V measurement complete.')   
        
    
    
    
    # --------------- functions for QCM ----------------------------------
    
    def get_harmonics(self):
        #get selected QCM harmonics from GUI checkboxes
        selected_harmonics = []
        if self.ui.n1_on.isChecked(): selected_harmonics.append(1)
        if self.ui.n3_on.isChecked(): selected_harmonics.append(3)
        if self.ui.n5_on.isChecked(): selected_harmonics.append(5)
        if self.ui.n7_on.isChecked(): selected_harmonics.append(7)
        if self.ui.n9_on.isChecked(): selected_harmonics.append(9)
        if self.ui.n11_on.isChecked(): selected_harmonics.append(11)
        if self.ui.n13_on.isChecked(): selected_harmonics.append(13)
        if self.ui.n15_on.isChecked(): selected_harmonics.append(15)
        print(selected_harmonics)
    
    
    
    #------- functions for pressure control ------------------------------
    
    def pressure_table_to_df(self):
        #convert pressure sequence table to pandas dataframe
        global pressure_df
        #create empty dataframe
        pressure_df = pd.DataFrame(columns=['time', 'rh'],
                                index=range(self.ui.pressure_table.rowCount()))
        #populate dataframe
        for rowi in range(self.ui.pressure_table.rowCount()):
            for colj in range(self.ui.pressure_table.columnCount()):
                new_entry = self.ui.pressure_table.item(rowi, colj).text()
                pressure_df.iloc[rowi, colj] = new_entry
        #delete empty rows
        pressure_df = pressure_df[pressure_df['time'] != '0']     
        return pressure_df

    def plot_pressure(self):
        #plot the pressure over time
        plt.ion
        fig_press = plt.figure(2)
        fig_press.clf()
        df = master_df[master_df['date'] != '0.0']
        
        plt.plot(df['time'].astype(float),
                 df['pressure'].astype(float),
                 c='k', lw=1)
        plt.xlabel('Elapsed time (min)', fontsize=fontsize)
        plt.ylabel('Pressure (Torr)', fontsize=fontsize)
        fig_press.canvas.set_window_title('Chamber pressure')
        plt.tight_layout()
        plt.draw()


# ----------------- functions for pressure sequence control ----------------

    def show_sequence_plot(self):
        #plot the pressure sequence
        try: 
            pressure_df = self.pressure_table_to_df()
            seq_time = np.array(pressure_df['time'].astype(float))
            plot_seq_time = np.insert(np.cumsum(seq_time), 0, 0)/60
            seq_rh = np.array(pressure_df['rh'].astype(float))
            plot_seq_rh = np.insert(seq_rh, 0, seq_rh[0])  

            fig_seq = plt.figure(1)
            plt.cla()
            plt.ion()
            plt.fill_between(plot_seq_time,
                             plot_seq_rh,
                             step='pre', alpha=0.6)
            plt.plot(plot_seq_time,
                     plot_seq_rh,
                     c='b', drawstyle='steps', alpha=0)
            plt.xlabel('Time (hours)', fontsize=fontsize)
            plt.ylabel('RH (%)', fontsize=fontsize)
            plt.tight_layout()
            fig_seq.canvas.set_window_title('Pressure sequence')
            fig_seq.show()
        except: self.ui.output_box.append('Pressure sequence not valid.')

    def clear_sequence(self):
        #clear pressure sequence
        #populate dataframe with 0's
        for rowi in range(self.ui.pressure_table.rowCount()):
            for colj in range(self.ui.pressure_table.columnCount()):
                self.ui.pressure_table.setItem( rowi, colj,
                        QtWidgets.QTableWidgetItem('0'))
        self.ui.output_box.append('Pressure sequence cleared.')

    def export_sequence(self):
        #save pressure sequence
        export_seq_name = QFileDialog.getSaveFileName(
                self, 'Create pressure sequence file to save',
                    '.csv')[0]
        pressure_df.to_csv(str(export_seq_name), index=False)       
        self.ui.output_box.append('Pressure sequence file exported.')

    def import_sequence(self):
        #import pressure sequence file
        import_seq_name = QFileDialog.getOpenFileName(
                self, 'Select pressure sequence file')[0]
        imported_seq = pd.read_csv(import_seq_name)
        #populate table on GUI
        for rowi in range(len(imported_seq)):
            for colj in range(len(imported_seq.columns)):
                self.ui.pressure_table.setItem(rowi, colj,
                    QtWidgets.QTableWidgetItem(str(imported_seq.iloc[rowi, colj])))
        self.ui.output_box.append('Pressure sequence file imported.')

    def stop_sequence(self):
        #send sequence early using when "STOP" sequence menu item is clicked
        self.stop_sequence = True
        
    def run_sequence(self):
        #run pressure sequence
        #disable other sequence options
        self.ui.output_box.append('Pressure sequence initiated.')
        self.ui.seq_clear.setDisabled(True)
        self.ui.seq_import.setDisabled(True)
        self.ui.seq_run.setDisabled(True)
        self.ui.seq_stop.setDisabled(False)
        self.ui.valve_mode.setDisabled(True)
        self.ui.pressure_mode.setDisabled(True)
        self.ui.set_pressure.setDisabled(True)
        self.ui.set_valve_position.setDisabled(True)
        self.ui.flow1.setDisabled(True)
        self.ui.flow2.setDisabled(True)
        self.ui.flow3.setDisabled(True)
        
        #set up timers and counters
        pressure_df = self.pressure_table_to_df().astype(float)
        elapsed_seq_time = 0
        seq_start_time = time.time()
        #loop over each step in sequence
        for step in range(len(pressure_df)):
            if self.stop_sequence == False:
                step_start_time = time.time()
                elapsed_step_time = 0
                step_dur = pressure_df['time'].iloc[step]
            else: break
            #repeat until step duration has elapsed
            while elapsed_step_time < step_dur:
                #use this to handle threading during the loop
                QtCore.QCoreApplication.processEvents()
                if self.stop_sequence == False:
                    #update step counters and timers on GU
                    elapsed_step_time = time.time() - step_start_time
                    self.ui.current_step_display.setText(str(int(step+1)))
                    self.ui.set_rh.setValue(pressure_df['rh'].iloc[step])
                    self.ui.elapsed_step_time_display.setText(
                            str(np.round(elapsed_step_time, decimals=3)))
                    elapsed_seq_time = time.time() - seq_start_time
                    self.ui.elapsed_seq_time_display.setText(
                            str(np.round(elapsed_seq_time, decimals=3)))
                    
                    # EXECUTE MEASUREMENTS INSIDE SEQUENCE HERE

                else: break
        self.stop_sequence = False                
        #reset sequence counters
        self.ui.current_step_display.setText('0')
        self.ui.elapsed_step_time_display.setText('0')
        self.ui.elapsed_seq_time_display.setText('0')
        self.ui.output_box.append('Pressure sequence completed.')
        #re-enable other sequence options
        self.ui.seq_clear.setDisabled(False)
        self.ui.seq_import.setDisabled(False) 
        self.ui.seq_run.setDisabled(False)
        self.ui.seq_stop.setDisabled(True)
        self.ui.valve_mode.setDisabled(False)
        self.ui.pressure_mode.setDisabled(False)
        self.ui.set_pressure.setDisabled(False)
        self.ui.set_valve_position.setDisabled(False)
        self.ui.flow1.setDisabled(False)
        self.ui.flow2.setDisabled(False)
        self.ui.flow3.setDisabled(False)
 




#--------------------------- run application ----------------------------
if __name__ == '__main__':
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else: app = QtWidgets.QApplication.instance() 
    window = App()
    window.update_fields()
    window.show()
    sys.exit(app.exec_())