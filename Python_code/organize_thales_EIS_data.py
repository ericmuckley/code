import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import colors, ticker, cm

def config_plot(xlabel='x', ylabel='y', size=16,
               setlimits=False, limits=[0,1,0,1]):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.rcParams.update({'figure.autolayout': True})
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    #set axis limits
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))


#%% import data

# filename with Bode plots
bodefile = r'C:\Users\a6q\exp_data\2019-07-02_ws2_5layer_rh_bodelist.csv' 
# read in file
bodedata = pd.read_csv(bodefile, skiprows = 16, skip_blank_lines=True,
                       error_bad_lines=False, warn_bad_lines=False, sep=',') 
# convert str to float, coerce to NaN, drop NaN
bodedata = bodedata.apply(pd.to_numeric, errors='coerce').dropna() 

# get length of a single spectrum
pointnum = len(np.unique(bodedata['Number'])) 
# retain only one array of frequencies
freq = np.array(bodedata['Frequency/Hz'].iloc[0:pointnum])

# convert spectra from columns to 2D arrays
ztot = np.reshape(np.array(bodedata['Impedance/Ohm']), (pointnum, -1), order='F')
phasedeg = np.reshape(np.array(bodedata['Phase/deg']), (pointnum, -1), order='F')
spectranum = len(ztot[0]) #number of spectra present

# delay in minutes between spectra measurements
delay = 10
time = (np.arange(spectranum)*delay + delay) / 60

# calculate Re(Z) and Im(Z) using phase information
phase = np.divide(np.multiply(np.pi, phasedeg), 180)
rez = np.multiply(ztot, np.cos(phase))
imz = np.multiply(ztot, np.sin(phase))





#%% plot results 
color_spec = plt.cm.seismic(np.linspace(0,1,spectranum))

#plot Bode Z data
[plt.loglog(freq, ztot[:,i], c=color_spec[i]) for i in range(spectranum)]
config_plot('Frequency (Hz)', 'Z ($\Omega$)')
plt.title('Total Z', fontsize=16)
plt.show()

#plot Bode phase data
[plt.semilogx(freq, phasedeg[:,i], c=color_spec[i]) for i in range(spectranum)]
config_plot('Frequency (Hz)', 'Phase (deg)')
plt.title('Phase', fontsize=16)
plt.show()

#create Nyquist plots
[plt.scatter(rez[:,i], imz[:,i], s=2) for i in range(spectranum)]
config_plot('Re(Z) ($\Omega$)', 'Im(Z) ($\Omega$)')
plt.title('Nyquist plots', fontsize=16)
plt.show()

#get z values at first frequency measured
highfreqz = ztot[0,:]
plt.plot(time, highfreqz, marker='o')
config_plot('Time (hrs)', 'Z @ first freq. ($\Omega$)')
plt.show()

#get z values at last frequency measured
lowfreqz = ztot[-1,:]
plt.plot(time[5:], lowfreqz[5:], marker='o')
config_plot('Time (hrs)', 'Z @ last freq. ($\Omega$)')
plt.show()

#%% format data into Origin-ready matrices
all_bode_z = freq
all_bode_phase = freq
all_nyquist = freq

for i in range(spectranum):
    all_bode_z = np.column_stack((all_bode_z, ztot[:,i]))
    all_bode_phase = np.column_stack((all_bode_phase, phasedeg[:,i]))
    all_nyquist = np.column_stack((all_nyquist, rez[:,i]))
    all_nyquist = np.column_stack((all_nyquist, imz[:,i]))

#%%plot heatmaps

#create x, y, and z points to be used in heatmap
xf = np.array(list(range(spectranum)))
yf = freq
Xf, Yf, Zf = np.array([]), np.array([]), np.array([])

#loop over each spectrum
for i in range(spectranum):
    #create arrays of X, Y, and Z values
    Xf = np.append(Xf, np.repeat(i, pointnum))
    Yf = np.append(Yf, freq)
    Zf = np.append(Zf, phasedeg[:,i])

zf = griddata((Xf, Yf), Zf, (xf[None,:], yf[:,None]), method='cubic')
#create the contour plot
CSf = plt.contourf(xf, yf, zf, 200,
                   cmap=plt.cm.seismic,
                   vmax=np.nanmax(Zf),
                   vmin=np.nanmin(Zf))
plt.yscale('log')
plt.colorbar()
config_plot('Spectra number', 'Frequency (Hz)')
plt.title('Phase (deg)', fontsize=16)
plt.show()


Xf, Yf, Zf = np.array([]), np.array([]), np.array([])
#loop over each spectrum
for i in range(spectranum):
    #create arrays of X, Y, and Z values
    Xf = np.append(Xf, np.repeat(i, pointnum))
    Yf = np.append(Yf, freq)
    Zf = np.append(Zf, ztot[:,i])
Zf = np.log(Zf)

zf = griddata((Xf, Yf), Zf, (xf[None,:], yf[:,None]), method='cubic')

#create the contour plot
CSf = plt.contourf(xf, yf, zf, 300, cmap=plt.cm.seismic,
                   vmax=np.nanmax(Zf), vmin=np.nanmin(Zf))
plt.yscale('log')
plt.colorbar()
config_plot('Spectra number', 'Frequency (Hz)')
plt.title('log(Z) ($\Omega$)', fontsize=16)
plt.show()
