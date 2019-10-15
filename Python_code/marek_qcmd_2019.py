# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# get list of excel files in folder ton convert from .xlsx to .csv
file_list = glob.glob(os.path.join(
    'C:\\Users\\a6q\\exp_data\\marek_casein_qcmd_2019', '*.xlsx'))
filename_list = [os.path.basename(file).split('.')[0] for file in file_list]

def convert_excel_to_csv(file_list):
    # converts a list of excel files to csv files. the excel files may have
    # multiple sheets, each of which is converted to an invidual csv file.
    # the final csv filenames are 'excelfilename_sheetname.csv'.
    # loop over each excel file in file_list
    for file_i, file in enumerate(file_list):
        print(os.path.basename(file))
        # open excel file to get the sheet names
        xl = pd.ExcelFile(file)
        sheet_names = xl.sheet_names
        # loop over all sheets in excel file
        for sheet_i, sheet in enumerate(sheet_names):
            print(sheet)
            # create new filename for the csv file
            #new_name = os.path.basename(file).split('.')[0]+'_'+sheet+'.csv'
            new_name = sheet+'.csv'
            print('new filename = '+new_name)
            
            # read dta from the excel file
            excel_data = pd.read_excel(file, sheet_name=sheet, header=None)
            # export the excel data to a csv file
            excel_data.to_csv(new_name)


#%%



def create_video(image_folder, video_name, fps=8, reverse=False):
    # create video out of images saved in a folder
    import cv2
    import os

    images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]
    if reverse:
        images = images[::-1]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, fps, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()


# %%

with open('exp_data\\marek_qcmd_2019_n_mag3.pkl', 'rb') as handle:
    notes_dict = pickle.load(handle)


# get list of csv files
file_list = glob.glob(os.path.join(
        'C:\\Users\\a6q\\exp_data\\marek_qcmd_csv_2019', '*.csv'))

# create empty dictionary to hold all formatted data
data_dict = {}
# create empty dictionary to hold largest magnitudes of formatted data
mag_dict = {}
# number of largest magnitudes to append to mag_dict
n_mags = 3
# parameters to place inside the data dictionary
params = ['freq', 'Q', 'D', 'R', 'G', 'L_mH', 'c_Pf']
harmonics = [1, 3, 5, 7, 9, 11]

# loop over each file
for file_i, file in enumerate(file_list):
    sheet_name = os.path.basename(file).split('.')[0]
    print('processing sheet %i / %i (name: %s)' % (file_i + 1,
                                                  len(file_list),
                                                  sheet_name))
    # read in data
    raw_data = pd.read_csv(file)
    # remove unwanted index columns
    data = raw_data.drop(columns=['Unnamed: 0', '0', '10', '11'])
    # rename columns
    data.rename(columns={'1': 'time', '2': 'n', '3': 'freq',
                         '4': 'Q', '5': 'D', '6': 'R', '7': 'G',
                         '8': 'L_mH', '9': 'c_Pf'}, inplace=True)

    # get total number of samples and harmonics in each file
    num_of_samples = len(data[data['n'] == 1])
    tot_harmonics = len(np.unique(data['n']))
    colormap = plt.cm.rainbow(np.linspace(0, 1, tot_harmonics))[::-1]

    # create empty array to hold time and harmonic-dependent frequency
    data_dict[sheet_name] = np.zeros((num_of_samples,
                                         tot_harmonics + 1, len(params)))
    mag_dict[sheet_name] = np.zeros((n_mags * tot_harmonics, len(params)))

    # populate data dictionary by looping over each measured parameter
    for param_i, param in enumerate(params):

        # populate time column
        data_dict[sheet_name][:, 0, param_i] = data[
                'time'][:num_of_samples]/3600

        # populate data columns by looping over each harmonic
        for n_i, n in enumerate(np.unique(data['n'])):

            # populate data column
            data_dict[sheet_name][:, n_i + 1, param_i] = data[
                    data['n'] == n][param] - data[data['n'] == n][param].iloc[0]
            
            # find "n_mags" number of largest abs magnitudes in data column
            abs_mags = np.abs(data_dict[sheet_name][:, n_i + 1, param_i])
            param_max_ind = np.argsort(abs_mags)[::-1][:n_mags]
            param_max = data_dict[sheet_name][param_max_ind, n_i + 1, param_i]
            mag_dict[sheet_name][n_i*n_mags:n_i*n_mags+n_mags, param_i] = param_max
        
            if param == 'D' or param == 'freq':
                # plot delta f vs time
                plt.plot(data_dict[sheet_name][:, 0, param_i],
                data_dict[sheet_name][:, n_i + 1, param_i], label=n,
                         c=colormap[n_i])


        if param == 'D' or param == 'freq':
            plt.title(sheet_name, fontsize=16)
            plt.legend(fontsize=8, ncol=2)
            plt.ylabel('Delta '+param, fontsize=16)
            plt.xlabel('Time (hours)', fontsize=16)
            plt.tight_layout()
            save_pic_filename = 'exp_data\\casein_plots0\\fig'+str(
                    file_i).zfill(3)+'.jpg'
    
            # plt.savefig(save_pic_filename, format='jpg', dpi=250)
            plt.gcf().set_size_inches(4, 3)
            plt.show()


with open('exp_data\\marek_qcmd_2019.pkl', 'wb') as p:
    pickle.dump(data_dict, p, protocol=pickle.HIGHEST_PROTOCOL)




# %% create dataframe to hold all magnitude data
'''
# read in sheet which decribes each experiment and create a dictionary
experiment_notes_df = pd.read_csv(
        'C:\\Users\\a6q\\exp_data\\marek_casein_2019_notes.csv')
notes_dict = dict(zip(experiment_notes_df.filename,
                      experiment_notes_df.description))

# construct data array to place inside dataframe
mag_arr = np.zeros((0, len(params)+3))  # len(params) + n + file + note
key_arr = []
note_arr = []

for key in mag_dict:
    # create arrays with file names and descirptions
    key_arr = np.append(key_arr, np.full(n_mags*6, key))
    note_arr = np.append(note_arr, np.full(n_mags*6, notes_dict[key]))
    # create array to hold all the magnitude data
    new_arr = np.zeros((len(harmonics)*n_mags, len(params)+3))
    new_arr[:, 2] = np.repeat(harmonics, n_mags)
    new_arr[:, 3:] = mag_dict[key]
    mag_arr = np.vstack((mag_arr, new_arr))

# assemble all arrays into a single dataframe 
mag_df = pd.DataFrame(columns=['note', 'file', 'n', 'freq', 'Q', 'D',
                               'R', 'G', 'L_mH', 'c_Pf'], data=mag_arr)
mag_df['file'] = key_arr
mag_df['note'] = note_arr

with open('exp_data\\marek_qcmd_2019_n_mag3.pkl', 'wb') as p:
    pickle.dump(mag_df, p, protocol=pickle.HIGHEST_PROTOCOL)
'''

# %% plot largest magnitude of parameter vs another

# parameters to plot
x_var = 0
y_var = 2  

'''
for sheet_i, sheet in enumerate(mag_dict):
    print('processing file %i / %i' % (sheet_i + 1, len(data_dict)))
    x  = mag_dict[sheet][1:, x_var] * 1/np.array(harmonics)[1:]
    plt.plot(x)
plt.show()
'''


'''
for sheet_i, sheet in enumerate(mag_dict):
    print('processing file %i / %i' % (sheet_i + 1, len(data_dict)))
    # loop over each harmonic
    for row_i in range(len(mag_dict[sheet])):
        x  = mag_dict[sheet][row_i, x_var] / harmonics[row_i]
        y = mag_dict[sheet][row_i, y_var] 
        plt.scatter(x, y, s=row_i*5)#, label=n)#, c=colormap[n_i])
    plt.title(sheet, fontsize=16)
    #plt.legend(fontsize=8, ncol=2)
    plt.ylabel(params[y_var], fontsize=16)
    plt.xlabel(params[x_var], fontsize=16)
    plt.tight_layout()
    save_pic_filename = 'exp_data\\casein_plots0\\fig'+str(
            file_i).zfill(3)+'.jpg'

    # plt.savefig(save_pic_filename, format='jpg', dpi=250)
    plt.gcf().set_size_inches(8, 6)
plt.show()
'''

# %% plot one parameter vs another

# parameters to plot
x_var = 1
y_var = 4  
    
'''  
for sheet_i, sheet in enumerate(data_dict):
    print('processing file %i / %i' % (sheet_i + 1, len(data_dict)))
    # loop over each harmonic
    for n_i, n in enumerate(np.unique(data['n'])):
        
        # don't include first harmonic because of noise
        if n != 1:
            x_arr  = data_dict[sheet][:, n_i + 1, x_var]
            y_arr = data_dict[sheet][:, n_i + 1, y_var]
            
            plt.scatter(x_arr, y_arr, s=2, label=n)#, c=colormap[n_i])
                
    plt.title(sheet, fontsize=16)
    plt.legend(fontsize=8, ncol=2)
    plt.ylabel(params[y_var], fontsize=16)
    plt.xlabel(params[x_var], fontsize=16)
    plt.tight_layout()
    save_pic_filename = 'exp_data\\casein_plots0\\fig'+str(
            file_i).zfill(3)+'.jpg'

    # plt.savefig(save_pic_filename, format='jpg', dpi=250)
    plt.gcf().set_size_inches(4, 3)
    plt.show()
'''

# %% calculate total size of data dictionary

tot_points = 0
for key in data_dict:
    tot_points += data_dict[key].size
print('total points = %f million' % (tot_points/1e6))

# %% compile plots into video

make_video = False
if make_video:
    create_video('exp_data\\casein_plots0\\',
                 'C:\\Users\\a6q\\Desktop\\raw_df_casein_plots2.avi', fps=10)
