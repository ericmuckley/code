import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %% designate data folder name
data_folder_name = 'C:\\Users\\a6q\\exp_data\\2019-11-19_BB4'
n = 3

# %% define useful functions

# change matplotlib settings to make plots look nicer
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['ytick.major.width'] = 3


def plot_setup(labels=['X', 'Y'], fsize=18, setlimits=False,
               limits=[0, 1, 0, 1], title='', legend=True, save=False,
               filename='plot.jpg'):
    """Creates a custom plot configuration to make graphs look nice.
    This can be called with matplotlib for setting axes labels,
    titles, axes ranges, and the font size of plot labels.
    This should be called between plt.plot() and plt.show() commands."""
    plt.xlabel(str(labels[0]), fontsize=fsize)
    plt.ylabel(str(labels[1]), fontsize=fsize)
    plt.title(title, fontsize=fsize)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    if legend:
        plt.legend(fontsize=fsize-4)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()


def qcm_conductance(rs, xs):
    """Calculate conductance (G) of QCM spectrum using
    series resistance (Rs) and reactance (Xs)."""
    # complex impedance
    Z = np.add(rs, 1j*xs)
    # complex admittance
    Y = np.reciprocal(Z)
    # conductance
    G = np.real(Y)
    return G


def single_lorentz_peak(freq, Gp, Cp, Gmax, d0, f0):
    """Returns conductance spectrum of a single peak.
    Calculation is taken from Equation (2) in:
    Yoon, S.M., Cho, N.J. and Kanazawa, K., 2009.
    Analyzing spur-distorted impedance spectra for the QCM.
    Journal of Sensors, 2009.
    inputs:
    freq = frequency array
    Gp = conductance offset
    Cp = susceptance offset
    Gmax = maximum of conductance peak
    d0 = dissipation
    f0 = resonant frequency of peak (peak position)
    """
    # construct peak
    peak = Gmax / (1 + (1j/d0)*((freq/f0)-(f0/freq)))
    # add offsets to spectrum
    Y = Gp + 1j*2*np.pi*freq*Cp + peak
    G = np.real(Y)
    return G


def qcm_fit(f, rs, xs, plot=True):
    """Perform fit of QCM spectrum using frequency (f), series resistance
    (rs), and series reactance (xs)."""
    # get guess for resonant frequency
    f0 = f[np.argmax(rs)]
    # get frequency of highest reactance value
    f0_xs = f[np.argmax(xs)]
    # estimate FWHM of the resonant peak
    fwhm = 2 * np.abs(f0 - f0_xs)
    # create guess for dissipation
    d0 = fwhm / f0
    # ensure we don't try to divide by 0
    if d0 == 0:
        d0 = 1e-7

    # initial guess for the peak fit parameters: [Gp, Cp, Rs_max, d0, f0]
    guess = [np.min(rs), 0, np.max(rs), d0, f0]

    # perform the fit
    try:
        popt, _ = curve_fit(single_lorentz_peak, f, rs, p0=guess,
                            )#ftol=1e-12, xtol=1e-12)
        print('\nFIT SUCCESSFUL:')
        print('f0: %0.3e, D: %0.3e' % (f0, d0))
    except RuntimeError:
        print('\nFIT FAILED')
        popt = guess

    # extract fit parameters from fit
    f0, d0 = popt[4], np.abs(popt[3])*(1e6)
    fit = single_lorentz_peak(f, *popt)
    # plot spectrum and fit
    if plot:
        plt.scatter(f, rs, s=4, c='k', marker='o', alpha=0.3, label='data')
        plt.plot(f, fit, c='r', lw=0.75, label='fit')
        plot_setup(labels=['Frequency (Hz)', 'Rs (Ohm)'])
        plt.show()
    return f0, d0, fit


def get_n_from_filename(filename):
    """Get the harmonic number from a QCM spectra file using the filename.
    Returns an integer n and the column number in which n should be placed,
    assuming columns are ordered: [1, 3, 5, 7, 9, 11, 13, 15, 17]."""
    n = int(filename.split('__qcm_n=')[1].split('_spectra')[0])
    col_position = int((n - 1) / 2)
    return n, col_position


# %% organize files

start_time = time.time()

# get list of all files in data folder
filelist = glob.glob(data_folder_name + '/*')

# get list of all files which contain QCM spectra
qcm_spec_files = [f for f in filelist if 'qcm_n='+str(n) in f]

# get total_number of spectra
tot_spec_num = int(len(pd.read_csv(qcm_spec_files[0]).columns) / 3)


# select file which contains time information and get the times
time_file = [f for f in filelist if 'main_df' in f][0]
time_file = pd.read_csv(time_file).dropna(how='all')
times = time_file['time'].values.astype(float)
times -= times[0]
times = np.linspace(0, times[-1], num=tot_spec_num)

# initialize array to hold all fitting results
results = np.zeros((tot_spec_num, 19))
results[:, 0] = times







# %% loop over each spectra file and analyze the spectra

# loop over each file
for filename in qcm_spec_files:

    n, col_pos = get_n_from_filename(filename)
    data = pd.read_csv(filename)
    data = data.dropna(how='all')


    fits = np.zeros((len(data), 100))
    

    # loop over each spectrum in file
    for i in range(0, 60, 3):
        print('fitting n = %i, spectrum %i / %i' % (n, int((i/3)+1),
                                                    int(len(data.columns)/3)))
        f = data.iloc[:, i].values
        rs = data.iloc[:, i + 1].values
        xs = data.iloc[:, i + 2].values



        # perform fitting of QCM spectrum
        f0, d0, fit = qcm_fit(f, rs, xs, plot=False)
        results[int(i/3), col_pos + 1] = f0
        results[int(i/3), col_pos + 10] = d0
        
        
        
        fits[:, i] = f
        fits[:, i+1] = fit
        
        
        
        plt.plot(f, fit)

    plt.show()


        
        
        



# %% format fit results

# subtract each column from column's initial value to get "deltas"
for col in range(len(results[0])):
    results[:, col] = results[:, col] - results[0, col]

# create pandas dataframe to hold all results
results_df = pd.DataFrame(data=results,
                          columns=['time', 'df1', 'df3', 'df5', 'df7',
                                   'df9', 'df11', 'df13', 'df15', 'df17',
                                   'dd1', 'dd3', 'dd5', 'dd7', 'dd9',
                                   'dd11', 'dd13', 'dd15', 'dd17'])

# %% plot results

for i in range(1, 10):
    plt.plot(results[:, 0], results[:, i], label=str((i*2)+1))
plt.legend()
plt.title('delta F (Hz)')
plt.show()

for i in range(1, 10):
    plt.plot(results[:, 0], results[:, i + 9], label=str((i*2)+1))
plt.legend()
plt.title('delta D (x 10^-6)')
plt.show()


print('total time: %0.2f min' %((time.time() - start_time)/60))