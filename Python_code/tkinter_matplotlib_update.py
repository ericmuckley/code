# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:12:52 2018

@author: a6q
"""

import time, sys
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt




#define general functions
def label_axes(xlabel='x', ylabel='y', size=16):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)


#define GUI functions
def start_run_fun():
    """Enable scanning by setting the global flag to True."""
    global run
    run = True

def stop_run_fun():
    """Stop scanning by setting the global flag to False."""
    global run
    run = False

def quit_fun():
    stop_run_fun()
    global main_loop
    main_loop = False
    print('closing things now...')
    print('quitting now!')


#initialize loops
main_loop = True #main loop begins automatically 
run = False #run loop begins only after user input
main_i = 0  #main loop index
run_i = 0 #run loop index

#create app window
root = tk.Tk()
root.title('Real-time plotting GUI')
root.geometry('500x500')
app = tk.Frame(root)
app.grid()







#create button functionality
start_button = tk.Button(app, text='Start run', command=start_run_fun)
stop_button = tk.Button(app, text='Stop run', command=stop_run_fun)
quit_button = tk.Button(app, text='Quit app', command=quit_fun)



#inintialize figures
fig = plt.figure(1)
plt.ion()
canvas = FigureCanvasTkAgg(fig, master=root)
plot_widget = canvas.get_tk_widget()


#place buttons on GUI
start_button.grid()
stop_button.grid()
quit_button.grid()
plot_widget.grid()




#main loop
while main_loop == True:
    root.update()

    time.sleep(0.05)    
    main_i += 1
    
    
    if run:
        print('running...')
        
        time.sleep(0.1)
        
        
        xtest = np.arange(run_i)/2
        ytest = np.sin(xtest)+xtest/10
        
        plt.cla()
        plt.plot(xtest, ytest)
        label_axes('X', 'Y')
        plt.tight_layout()
        fig.canvas.draw()
        
        
        run_i += 1
        
        
    
    
    
#close program
root.update()
root.destroy()
exit

