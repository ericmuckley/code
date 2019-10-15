# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:12:52 2018

@author: a6q
"""
import os
import time, sys
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# create our class Window and inherit from the tkinter Frame class.
class Window(tk.Frame):
    # Define settings upon initialization
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)   
        #reference to the master widget, which is the tk window                 
        self.master = master
        #run init_window, which doesn't yet exist
        self.init_window()
        
    
    def init_window(self):
        #Creation of init_window
        self.master.title('test V0') #title of master widget
        self.pack(fill=tk.BOTH, expand=1) #give widget full width of window
        top_menu_bar = tk.Menu(self.master) #create a top toolbar menu
        self.master.config(menu=top_menu_bar)
        
        #create the file pulldown menus with various commands
        file_menu = tk.Menu(top_menu_bar, tearoff=0)
        file_menu.add_command(label='Quit', command=self.quit_fun)
        top_menu_bar.add_cascade(label='File', menu=file_menu)
        
        process_menu = tk.Menu(top_menu_bar, tearoff=0)
        process_menu.add_command(label='Run', command=run_process.start_run_fun)
        process_menu.add_command(label='Stop', command=run_process.stop_run_fun)
        top_menu_bar.add_cascade(label='Process', menu=process_menu)
        
        
    def quit_fun(self):
        #quit the program
        run_process.stop_run_fun()
        global main_loop
        main_loop = False
        print('Closing instruments...')
        print('Application successfully quit.')



class run_process():
    #class for running process which needs start and stop functionality
    def start_run_fun():
        """Enable scanning by setting the global flag to True."""
        global run
        run = True

    def stop_run_fun():
        """Stop scanning by setting the global flag to False."""
        global run
        run = False


#define general functions
def label_axes(xlabel='x', ylabel='y', size=16):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)



#%%







#initialize loops
main_loop = True #main loop begins automatically 
run = False #run loop begins only after user input
main_i = 0  #main loop index
run_i = 0 #run loop index

#create app window
root = tk.Tk()
#root.title('Real-time plotting GUI')
root.geometry('500x500')
app = Window(root)
app.grid()



#create button functionality
start_button = tk.Button(app, text='Start run',command=run_process.start_run_fun)
stop_button = tk.Button(app, text='Stop run', command=run_process.stop_run_fun)
#quit_button = tk.Button(app, text='Quit app', command=quit_fun)



#inintialize figures
fig = plt.figure(1)
plt.ion()
canvas = FigureCanvasTkAgg(fig, master=root)
plot_widget = canvas.get_tk_widget()


#place buttons on GUI
start_button.grid()
stop_button.grid()
plot_widget.grid()



def plot_fun():
    xtest = np.arange(run_i)/2
    ytest = np.sin(xtest)+xtest/10
    
    plt.cla()
    plt.plot(xtest, ytest)
    label_axes('X', 'Y')
    plt.tight_layout()
    fig.canvas.draw()
  
def print_fun():
    print('hi')
    root.after(100, print_fun)
    

#main loop
while main_loop == True:
    
    root.update()
    
    root.after(10)
    main_i += 1

    
    if run:
        print('running...')
        
        root.after(100)
        
        
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
