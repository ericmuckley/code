# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 01:34:03 2018

@author: Eric
"""
from tkinter.ttk import Progressbar
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
import pandas as pd
import winsound
from time import sleep
import numpy as np

window = Tk()
 
window.title("QCM Music")
 
window.geometry('350x400')
 


def import_file_click():
    #when the import file button is clicked
    
    #open file browse dialog box
    filename = filedialog.askopenfilename()

    #display selected filename
    filenamedisplay.configure(text=str(filename))

    #read file
    file = pd.read_table(filename)
    headersdisplay.configure(text=str(list(file)))


rh_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
rh_levels = np.arange(0, 100, 10)

tone_time = 100

def n_click():

    for i in range(len(rh_levels)):
        sleep(4*tone_time/1000)
        rhlabel.configure(text='RH = '+str(rh_levels[i])+ '%')
        bar['value'] = rh_levels[i]
        
        
        f0 = int(5e6*n.get() - 50 - rh_levels[i]*8 - 0.1*(n.get())**2)
        f0label.configure(text='Resonant frequency = '+format(f0)+' MHz')
        
        beep_freq = 900+200*n.get() - 7*rh_levels[i]
        print(beep_freq)
        
        winsound.Beep(beep_freq, tone_time)
        window.update()





#button for importing file
importbutton = Button(window, text="Import file", command=import_file_click)
importbutton.pack()#grid(column=0, row=0)

#filename label
filenamelabel = Label(window, text="File name:")
filenamelabel.pack()#grid(column=1, row=0)

#filename indicator
filenamedisplay = Label(window, text='No file')
filenamedisplay.pack()#grid(column=2, row=0)


#headers indicator
headersdisplay = Label(window)
headersdisplay.pack()#grid(column=5, row=1)


#Harmonic indicator label
nlabeldisplay = Label(window, text='Crystal harmonic number:')
nlabeldisplay.pack()#grid(column=0, row=3)

#radio buttons for harmonic number
n = IntVar()
rad1 = Radiobutton(window,text='1st', value=1, command=n_click, variable=n)
rad3 = Radiobutton(window,text='3rd', value=3, command=n_click, variable=n)
rad5 = Radiobutton(window,text='5th', value=5, command=n_click, variable=n)
rad7 = Radiobutton(window,text='7th', value=7, command=n_click, variable=n)
rad9 = Radiobutton(window,text='9th', value=9, command=n_click, variable=n)
rad_row = 4
rad1.pack()#grid(column=0, row=rad_row)
rad3.pack()#grid(column=1, row=rad_row)
rad5.pack()#grid(column=2, row=rad_row)
rad7.pack()#grid(column=3, row=rad_row)
rad9.pack()#grid(column=4, row=rad_row)


#RH level label
rhlabel = Label(window, text='RH = '+str(rh_levels[0])+ '%')
rhlabel.pack()#(column=0, row=7)
    

#RH level progress bar
style = ttk.Style()
style.theme_use('default')
style.configure("blue.Hoizontal.TProgressbar", background='black')
bar = Progressbar(window, length=200, style='blue.Horizontal.TProgressbar')
bar['value'] = 0
bar.pack()#grid(column=0, row=0)


#f0 label
f0label = Label(window, text="Resonant frequency = ")
f0label.pack()#grid(column=1, row=0)
 
window.mainloop()