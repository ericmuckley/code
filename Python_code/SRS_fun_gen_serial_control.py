# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:22:58 2020

@author: a6q
"""
import time
import serial
import numpy as np
from serial.tools import list_ports

def print_ports():
    """Print a list of avilable serial ports."""
    ports = list(list_ports.comports())
    print('Available serial ports:')
    [print(p.device) for p in ports]

def open_port(address):
    """Open serial port using port address, e.g. 'COM6'."""
    return serial.Serial(port=address, timeout=2)



#print_ports()


dev = open_port('COM6')
dev.write('*IDN?\r'.encode())
print(dev.readline())




# set pulse width in seconds
da = 0
db = 1e-3
ampl_a = 1
pulse_spacing = 1e-6


# set trigger source to single shot trigger
dev.write('TSRC5\r'.encode())
# set delay of A and B outputs
dev.write(('DLAY2,0,'+str(da)+'\r').encode())
dev.write(('DLAY3,2,'+str(da + db)+'\r').encode())
# set amplitude of output A
dev.write(('LAMP1,'+str(np.clip(ampl_a,0.5,5))+'\r').encode())


def trigger_pulses(n, pulse_spacing):
    """Fire a single burst of n pulses with spacing in seconds."""
    for _ in range(n):
        # initiate single shot trigger
        dev.write('*TRG\r'.encode())
        time.sleep(pulse_spacing)



# set front panel display to show AB pulse amplitude
#dev.write('DISP12,3\r'.encode())


dev.close()
