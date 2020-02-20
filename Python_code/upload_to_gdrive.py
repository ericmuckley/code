# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:36:14 2020
@author: ericmuckley@gmail.com

MUST HAVE client_secrets.json FILE IN THE DIRECTORY
"""
import time

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def gdrive_connect():
    """Set up authentication for Google drive. This will not work
    unless the credential file 'client_secrets.json' is sitting in the
    same directory."""
    # get Google app credentials
    gauth = GoogleAuth() 
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    # return the google drive instance
    return drive

def gdrive_upload(drive, filename, foldername):
    """Upload a local file to a folder in Google drive. This will not work
    unless the credential file 'client_secrets.json' is sitting in the
    same directory."""
    # iterate through all folders to get folder ID number
    folders = drive.ListFile({
            'q': "'root' in parents and trashed=false"}).GetList()
    fid = [f['id'] for f in folders if f['title'] == foldername]
    # upload file
    f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": fid}]})
    f.SetContentFile(filename)
    f.Upload()
    

# set file to upload, and folder in Google drive in which to upload it
filename = r'C:\Users\a6q\EricMuckley_Citrine_challenge.ipynb'
folder_name = 'automated_uploads'

# connect to google drive
drive = gdrive_connect()


for i in range(3):
    
    # upload file to google drive
    gdrive_upload(drive, filename, folder_name)
    time.sleep(2)
















