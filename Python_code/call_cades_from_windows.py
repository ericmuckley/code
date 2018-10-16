# -*- coding: utf-8 -*-

#%% instructions for use 

#- open PuTTYgen
#- load an existing private PuTTY key
#- go to conversions --> export OpenSSH key and save it as a .pem file
#- set diectory of OpenSSH key on local machine in this code in 'k' variable
#- set CADES IP address in hostname argument of ssh.connect command
#- set directories of data files to call


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

def run_script_on_cades(ssh, cades_script_path):
    #run 'cade_script_path' python script on CADES using open ssh connection
    stdin, stdout, stderr = ssh.exec_command('python '+cades_script_path)
    [print(line) for line in stdout.readlines()]
    [print(line) for line in stderr.readlines()]

#%%

key_file = 'C:\\Users\\a6q\\tf-container.pem'
cades_IP = '172.22.5.231'

local_file = 'C:\\Users\\a6q\\call_windows_from_cades.py'
cades_file = '/home/cades/lab_comm/call_windows.py'

#open ssh connection to CADES
ssh = open_cades_ssh(cades_IP, key_file)

#transfer file to CADES
windows_to_cades(ssh, local_file, cades_file)

#run script on CADES
#run_script_on_cades(ssh, cades_file)









