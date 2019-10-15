# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:12:52 2018

@author: a6q
"""
import time
import numpy as np
import tkinter as tk
import tkinter.scrolledtext as tkst
from tkinter import filedialog as tkfiledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd


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
        self.master.title('Real-time Python Plotting GUI') #title of master widget
        self.pack(fill=tk.BOTH, expand=1) #give widget full width of window
        top_menu_bar = tk.Menu(self.master) #create a top toolbar menu
        self.master.config(menu=top_menu_bar)
        
        #create the file pulldown menus with various commands
        file_menu = tk.Menu(top_menu_bar, tearoff=0)
        file_menu.add_command(label='Quit', command=self.quit_fun)
        top_menu_bar.add_cascade(label='File', menu=file_menu)
        
        process_menu1 = tk.Menu(top_menu_bar, tearoff=0)
        process_menu1.add_command(label='Run', command=process1.start)
        process_menu1.add_command(label='Stop', command=process1.stop)
        process_menu1.add_command(label='Clear data', command=process1.clear)
        top_menu_bar.add_cascade(label='Process 1', menu=process_menu1)
        
        
        process_menu2 = tk.Menu(top_menu_bar, tearoff=0)
        process_menu2.add_command(label='Run', command=process2.start)
        process_menu2.add_command(label='Stop', command=process2.stop)
        top_menu_bar.add_cascade(label='Process 2', menu=process_menu2)
        
        
        
    def quit_fun(self):
        #stop the process, then quit the program
        process1.stop()
        print('Closing instruments...')
        print('Application successfully quit.')
        root.destroy()
        exit






class process1():
    #class for running process which needs start and stop functionality
    run = False
    def start():
        """Enable scanning by setting the global flag to True."""
        global run
        process1.run = True
    def stop():
        """Stop scanning by setting the global flag to False."""
        global run
        process1.run = False
    def clear():
        global data_arr1
        data_arr1 = np.empty((0,4))

class process2():
    #class for running process which needs start and stop functionality
    run = False
    def start():
        """Enable scanning by setting the global flag to True."""
        global run
        process2.run = True
    def stop():
        """Stop scanning by setting the global flag to False."""
        global run
        process2.run = False




#define general functions
def label_axes(xlabel='x', ylabel='y', size=16):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)



def increase_loop_counters():
    #increase main loop counter
    global main_i
    global process1_i
    global process2_i
    
    #increase loop counters
    main_i += 1
    if process1.run == True:
        process1_i += 1
    if process2.run == True:
        process2_i += 1
    
    #update coiunter labels on GUI
    main_i_display.configure(text='Main counter: '+str(main_i))
    process1_i_display.configure(text='Process 1 counter: '+str(process1_i))  
    process2_i_display.configure(text='Process 2 counter: '+str(process2_i)) 
    root.after(process_wait, increase_loop_counters)



def data_acquisition():
    #acquire data and plot in realtime
    global data_arr1    
    if process1.run == True:
        elapsed_time = (time.time() - start_time)/60
        
        #acquire new data and mark whether it should be saved or not
        new_data = np.array([elapsed_time,
                             process1_i,
                             np.random.random(),
                             save_now.get()])
        #store data
        data_arr1 = np.vstack((data_arr1, new_data))
        #plot data
        plt.cla()
        plt.plot(data_arr1[:,1], data_arr1[:,2])
        label_axes('X', 'Y')
        plt.title('Process 1, plot '+str(main_i), fontsize=16)
        plt.tight_layout()
        fig1.canvas.draw() 

    root.after(process_wait, data_acquisition)


def save_data_fun():
    #write data to file
    global save_filename
    
    if process1.run == True: #if process is running
        if save_now.get(): #if save checkbox is activated
            save_df = pd.DataFrame(data=data_arr1,
                        columns=['minutes', 'X','Y', 'SAVE?'])
            
            try: #save data
                save_df.to_csv(save_filename.name, sep='\t', index=False)
            except: #if data file doesn't exist, create it
                save_filename = tkfiledialog.asksaveasfile(mode='w', defaultextension='.txt')
                save_df.to_csv(save_filename.name, sep='\t', index=False)
            save_file_label.configure(text='Saving to: '+str(save_filename.name))
        else: #if save checkbox is not activated
            save_file_label.configure(text='NOT SAVING')

    root.after(process_wait, save_data_fun) 
  
    
'''
def clear_data():
    global data_arr1
    data_arr1 = np.empty((0,4))
'''
def write_to_textbox():
    textbox.insert(tk.END, 'loop iteration '+str(main_i)+'\n')
    textbox.see('end')
    
def clear_textbox():
    textbox.delete('1.0', tk.END)   


def print_v_steps():
    all_steps = np.linspace(float(start_v.get()),
                            float(end_v.get()),
                            int(steps.get()))
    textbox.insert(tk.END, 'user-defined voltages = '+str(all_steps)+'\n')
    textbox.see('end')
    


#%%

#create app window
root = tk.Tk()
#root.title('Real-time plotting GUI')
root.geometry('1000x700')
app = Window(root)
app.grid()


#initialize variables
main_i = 0  #main loop index
process1_i = 0 #run loop index
process2_i = 0 #run loop index
process_wait = 100
start_time = time.time()
data_arr1 = np.empty((0,4))
save_filename = None
select_harmonics = {}
for i in range(1, 13, 2):
    select_harmonics[str(i)]=False


#inintialize figures
fig1 = plt.figure(1)
plt.ion()
plot1 = FigureCanvasTkAgg(fig1, master=root).get_tk_widget()


#counters frame
frame_counters = tk.LabelFrame(root, text='Counters')
frame_counters.grid(column=0, row=0, rowspan=2, sticky='NW', padx=5, pady=5)
main_i_display = tk.Label(frame_counters, text='Main counter: '+str(0))
main_i_display.grid(row=1, column=0, sticky='W', pady=0)
process1_i_display = tk.Label(frame_counters, text='Process 1 counter: '+str(0))
process2_i_display = tk.Label(frame_counters, text='Process 2 counter: '+str(0))
process1_i_display.grid(row=2, column=0, sticky='W', pady=0)
process2_i_display.grid(row=7, column=0, sticky='W', pady=0)

#saving frame
frame_saving =  tk.LabelFrame(root, text='Saving')
frame_saving.grid(column=0, row=2, sticky='W', padx=5, pady=1)
save_now = tk.BooleanVar()
button_save_now = tk.Checkbutton(frame_saving, text='Save data', variable=save_now)
save_file_label = tk.Label(frame_saving, text='Save file: NO FILE')
button_save_now.grid(row=1, column=1, sticky='W')
save_file_label.grid(row=2, column=1, sticky='W')



#input frame
frame_input = tk.LabelFrame(root, text='Scan settings')
frame_input.grid(column=1, row=0, rowspan=4, columnspan=2, sticky='NW', padx=5, pady=5)
start_label = tk.Label(frame_input, text='Start V')
start_label.grid(row=0, column=0, padx=0, pady=2, sticky='W')
start_v = tk.StringVar()
start_v.set('0')
start_field = tk.Entry(frame_input, width=8, textvariable=start_v) 
start_field.grid(row=0, column=1, padx=2, pady=2, sticky='W')
end_label = tk.Label(frame_input, text='End V')
end_label.grid(row=1, column=0, padx=0, pady=2, sticky='W')
end_v = tk.StringVar()
end_v.set('1')
end_field = tk.Entry(frame_input, width=8, textvariable=end_v) 
end_field.grid(row=1, column=1, padx=2, pady=2, sticky='W')
steps_label = tk.Label(frame_input, text='Steps')
steps_label.grid(row=2, column=0, padx=0, pady=2, sticky='W')
steps = tk.StringVar()
steps.set('11')
steps_field = tk.Entry(frame_input, width=8, textvariable=steps) 
steps_field.grid(row=2, column=1, padx=2, pady=2, sticky='W')
button_apply_input = tk.Button(frame_input, text='Apply input', command=print_v_steps)
button_apply_input.grid(row=3, column=0, columnspan=2, padx=2, pady=2, sticky='E')



#harmonics frame
frame_harmonics = tk.LabelFrame(root, text='Harmonics')
frame_harmonics.grid(column=3, row=0, sticky='NW', rowspan=6, padx=5, pady=5)
for i in range(1, 13, 2):
    select_harmonics[str(i)] = tk.Variable()
    harmonic_box = tk.Checkbutton(frame_harmonics,
                                  text=str(i),
                                  variable=select_harmonics[str(i)])
    harmonic_box.grid(row=i, column=1, padx=5, pady=2, sticky='W')




#output frame
frame_output = tk.LabelFrame(root, text='Output')
frame_output.grid(column=0, row=5, sticky='W', columnspan=6, padx=5, pady=5)
button_add_text = tk.Button(frame_output, text='Add text', command=write_to_textbox)
button_add_text.grid(row=0, column=0, padx=2, pady=2, sticky='W')
button_clear_text = tk.Button(frame_output, text='Clear text', command=clear_textbox)
button_clear_text.grid(row=0, column=1, padx=2, pady=2, sticky='W')
textbox = tkst.ScrolledText(frame_output, height=10, width=80)
textbox.grid(row=1, column=0, sticky='W', columnspan=8, padx=5, pady=5)

plot1.grid(row=0, column=4, rowspan=4, columnspan=6, sticky='W', padx=5, pady=5)

#run recurring functions
increase_loop_counters()
save_data_fun()
data_acquisition()

root.mainloop()   
