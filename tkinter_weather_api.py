# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:35:16 2020

@author: a6q
"""

from tkinter import *
import requests

win = Tk()
win.title("vole's weather app")
label1 =Label(win,text = "vole weather forecast",width =40,height = 3,bg = "black",fg = "red")
label1.pack()
label2 = Label(win,text="enter the city",bg="grey",fg =  "orange").place(x=30,y=68)
e = Entry(win,width =30)
e.place(x=151,y=65)
e.get()

def request():

    enter = input("enter your city :")
    url = "http://api.openweathermap.org/data/2.5/weather?q={}&APPID=f58dc36e5267cb72d14088861a4bbd75".format(enter)
    res = requests.get(url)

    data  = res.json()

    temp = data['main']['temp']
    print('Temperature : {}degree celcius'.format(temp))


    e.get()
request()