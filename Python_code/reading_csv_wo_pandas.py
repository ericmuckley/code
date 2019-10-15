# -*- coding: utf-8 -*-
"""
Im,orting and formatting a CSV file without use of Pandas

Created on Fri Mar  8 10:40:06 2019
@author: ericmuckley@gmail.com
"""
import os
import csv
import glob



def get_file_dict(exp_start_time, data_folder):
    # get a dictionary of each data file and what type of data it holds based
    # on the experiment start time and folder of data files.
    # get list of all data files in data folder
    all_data_files = glob.glob(data_folder + '\\' + '\*')
    # list of data descriptors (strings) which should show up in file names
    data_file_descriptors = ['main', 'qcm_params', 'iv', 'cv', 
                             'bs', 'optical']
    # create empty dictionary to hold selected data files
    file_dict = {}
    # loop through all data files in the data folder
    for f in all_data_files:
        # split full file path into file directory and name 
        filedir, filename = os.path.split(f)
        # get date of file creation from beginning of filename
        filedate = filename.split('__')[0]
        # select files which creation date matches that of exp_start_time
        if filedate == exp_start_time:
            # assign each datafile to each file descriptor
            for descriptor in data_file_descriptors:
                if descriptor in filename:
                    file_dict[descriptor] = f
    return file_dict


def import_csv(filename):
    # imports a csv file and returns headers and data as floats
    data = []
    with open(filename) as f:
        reader = csv.reader(f)
        [data.append(row) for row in reader]
    # extact headers
    headers = data[0]
    # remove headers from data
    data = data[1:]
    # convert strings to floats            
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == '':
                data[i][j] = '0'
            data[i][j] = float(data[i][j])
    # remove completely empty rows from data
    data = [row for row in data if not all(i == 0 for i in row)]
    return headers, data




# path to data_folder
data_folder = 'C:\\Users\\a6q\\exp_data\\sample_imes_python_data'
# experiment start time
exp_start_time = '2019-03-05_17-56'


file_dict = get_file_dict(exp_start_time, data_folder)

data_dict = {}
header_dict = {}

for file in file_dict:
    filename = file_dict[file]
    print(file)
    
    if file != 'main' and file != 'qcm_params':
        headers, data = import_csv(filename)
        data_dict[file] = np.array(data)
        header_dict[file] = headers
        print(len(data))
        print(len(data[0]))

        