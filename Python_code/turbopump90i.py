# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:38:35 2019

@author: ericmuckley@gmail.com
"""

# controlling leybold vacuum turbovac 90i turbo pump

import serial
import sys
import glob
import time

def list_serial_ports():
    # lists available serial ports
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes the current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')
    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


if __name__ == '__main__':
    print(list_serial_ports())
    
    turbo = serial.Serial(port='COM6', baudrate=19200, timeout=5)

    print('Turbo pump serial port is opened: '+str(turbo.isOpen()))
    
    turbo.write('02'.encode())
    #turbo.write('IND1'.encode())
    #turbo.write('PKE176'.encode())
    #turbo.flush()
    time.sleep(1)
    message = turbo.readline()
    
    print(message)
    
    
    turbo.close()