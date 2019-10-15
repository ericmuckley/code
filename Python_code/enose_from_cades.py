# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:49:30 2018
@author: ericmuckley@gmail.com
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

fontsize=16
plt.rcParams['xtick.labelsize'] = fontsize 
plt.rcParams['ytick.labelsize'] = fontsize


#%% open SSH connection to CADES
def open_cades_ssh(cades_IP, key_file):
    #open a connection to CADES using CADES IP address and the key file 
    import paramiko
    k = paramiko.RSAKey.from_private_key_file(key_file)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_system_host_keys()
    ssh.connect(hostname=cades_IP, username='cades', pkey=k)
    return ssh

def windows_to_cades(ssh, local_file, cades_file):
    #send local_file' (#original file on windows) to
    # 'cades_file' (destination file on cades) using open ssh connection
    ftp = ssh.open_sftp()
    ftp.put(local_file, cades_file)
    ftp.close()

def get_from_cades(ssh, local_file, cades_file):
    #send local_file' (#original file on windows) to
    # 'cades_file' (destination file on cades) using open ssh connection
    ftp = ssh.open_sftp()
    ftp.get(cades_file, local_file)
    ftp.close()

def run_script_on_cades(ssh, cades_script_path):
    #run 'cade_script_path' python script on CADES using open ssh connection
    stdin, stdout, stderr = ssh.exec_command('python '+cades_script_path)
    [print(line) for line in stdout.readlines()]
    [print(line) for line in stderr.readlines()]


#%% setup connection to cades
key_file = 'C:\\Users\\a6q\\tf-container.pem'
cades_IP = '172.22.5.231'
local_file = 'C:\\Users\\a6q\\exp_data\\enose_from_cades.pkl'
cades_file = '/home/cades/rpi_transfer_complete.pkl'


#%% plot data

#number of most recent points to show in plot
recent_points = 1e6
#plot every nth point
nth_point = 10

for i in range(1):
    
    #open ssh connection to CADES
    ssh = open_cades_ssh(cades_IP, key_file)
    #transfer data file from CADES
    get_from_cades(ssh, local_file, cades_file)
    
    #try to import data, if file is busy then wait
    try:
        #import data file
        df = pd.read_pickle(local_file)
        print('data file contains %i channels with %i points each' %(
                len(df.columns), len(df)))
        print('displaying most recent %i points' %recent_points)
        df = df.iloc[-recent_points::nth_point]
        

        
        #set up figure
        fig, (ax_temp, ax_press, ax_rh, ax_mq) = plt.subplots(4, sharex=True)
        
        
        #set x value for plotting: elapsed time in minutes or date and time
        x_axis = 'date-time' #options: 'date-time', 'time elapsed'
        
        if x_axis == 'time elapsed':
            x = df['time_elapsed'].astype(float)
            ax_mq.set_xlabel('Elapsed time (min)', fontsize=fontsize)
            
        elif x_axis == 'date-time':
            x = df['time']
            x = [datetime.fromtimestamp(
                    time.mktime(time.strptime(x0))) for x0 in x]
            plt.xticks(rotation=90)
            ax_mq.set_xlabel('Date', fontsize=fontsize)
            
            
        else:
            print('invalid x-axis type')
            break
        

        
        
        #plot temp, pressure, rh
        ax_temp.plot(x, df['temp'].astype(float), c='r')
        ax_press.plot(x, df['press'].astype(float), c='g')
        ax_rh.plot(x, df['rh'].astype(float), c='b')
    
        #plot MQ-X response
        #for ch in range(8):
        [ax_mq.plot(x,
                    df[df.columns[ch+5]].astype(float) - df[
                            df.columns[ch+5]].astype(float)[0],
                    label=df.columns[ch+5]) for ch in range(8)]
        ax_mq.legend(fontsize=10)#, loc='upper left')
        
        #label axes
        ax_mq.set_ylabel('MQ-X ($\Delta$V)', fontsize=fontsize)
        ax_temp.set_ylabel('Temp (C)', fontsize=fontsize)
        ax_press.set_ylabel('Press. (mbar)', fontsize=fontsize)
        ax_rh.set_ylabel('RH (%)', fontsize=fontsize)
        
        

        
        
        #configure plot
        plt.subplots_adjust(left=None, bottom=None,
                            right=None, top=None, 
                            wspace=None, hspace=0.05)
        fig.set_size_inches((10,12))

        plt.show()
        
        
        print(time.ctime())
        time.sleep(2)
        
    except:
        time.sleep(2)

























