# -*- coding: utf-8 -*-
"""
Assumes Flow Connector unit IDs are A, B, C, etc. 
"""
from alicat import FlowController
import serial.tools.list_ports

#Automatically detects COM ports - MUST USE Multi-Drop
#Controller unit IDs must be 'A', 'B', 'C', ...
#Returns dictionary of Flow Controller objects:
#   {'A':FlowControllerA, 'B':FlowControllerB, ....}



alicat_ports = serial.tools.list_ports.grep("067B:2303")    
controller = {}
unit_id = 'A'
#Loops through all connected USB ports (will only be one if RS232 Multi-Drop is connected)
for port in alicat_ports:
    alicat_COM_port = port.device
    next_port_exists = True
    
    #Checks if a next Unit ID exists, then adds to controller object controller dictionary
    while(next_port_exists):
        flow_controller = FlowController(port=alicat_COM_port, address=unit_id)
        if(flow_controller.is_connected(alicat_COM_port, address=unit_id) == True):
            #adds controller object to dictionary
            controller[unit_id] = flow_controller
            #gets next letter
            unit_id = chr(ord(unit_id) + 1) 
        else:
            next_port_exists = False




'''
def get_flow_controllers():
    #https://pyserial.readthedocs.io/en/latest/tools.html
    #Searches for ALICAT COM Port using its USB Product ID: 067B:2303
    alicat_ports = serial.tools.list_ports.grep("067B:2303")    
    controller = {}
    unit_id = 'A'
    #Loops through all connected USB ports (will only be one if RS232 Multi-Drop is connected)
    for port in alicat_ports:
        alicat_COM_port = port.device
        next_port_exists = True
        
        #Checks if a next Unit ID exists, then adds to controller object controller dictionary
        while(next_port_exists):
            flow_controller = FlowController(port=alicat_COM_port, address=unit_id)
            if(flow_controller.is_connected(alicat_COM_port, address=unit_id) == True):
                #adds controller object to dictionary
                controller[unit_id] = flow_controller
                #gets next letter
                unit_id = chr(ord(unit_id) + 1) 
            else:
                next_port_exists = False
    
    return controller
'''
'''
controller = {}
controller = get_flow_controllers()
print(controller['A'].get())
print(controller['B'].get())
controller['A'].set_gas('Air')
controller['B'].set_gas('Air')
for key in controller:
    controller[key].close()
'''
   
   
'''
#example usage
from alicat import FlowController

flow_controller = FlowController(port='COM4', address='A')
flow_controller.set_gas('Air')
print(flow_controller.get())
#print(flow_controller._get_control_point())
#flow_controller._set_setpoint(setpoint, retries=2)
print(flow_controller.is_connected('COM4', address='A'))

flow_controller.close()
'''

