

#%% open SSH connection to CADES

#import libs.ssh_connect as ssh

#%% open ssh connection to windows



cades_IP = '172.22.5.231' #IP address of tf-container CADES MV
desktop_IP = '128.219.194.40' #IP address of cubicle esktop computer
lab_IP = '160.91.17.159' #IP address of lab laptop

username = 
password = 


windows_IP = desktop_IP




import paramiko




try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(windows_IP, username=username, password, timeout=10)
    print('Connected')
except paramiko.AuthenticationException:
    print('Failed to connect due to wrong username/password')
    exit(1)
except Exception as e:
    print(e.message)    
    exit(2)





#run script
#stdin, stdout, stderr = ssh_obj.exec_command('ipconfig') 
#%%







'''

ssh_obj = ssh.connect_to_windows(lab_laptop_IP, lab_username, lab_password)

stdin, stdout, stderr = ssh_obj.exec_command("dir")


 open SSH connection to CADES

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
'''