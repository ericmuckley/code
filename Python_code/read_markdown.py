# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:01:38 2020

@author: a6q
"""

#%%
import markdown
import webbrowser


def fun(y=10):
    in_filename = r'C:\Users\a6q\Documents\GitHub\laser_triggering\README.md'
    out_filename = r'C:\Users\a6q\Documents\GitHub\laser_triggering\README.html'
    markdown.markdownFromFile(input=in_filename, output=out_filename)
      
    
    webbrowser.open(out_filename)

fun()