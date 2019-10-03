import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

def label_axes(xlabel='x', ylabel='y', size=16):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)

def singleBvD_reY(freq, Gp, Cp, Gmax00, D00, f00):
    # Returns admittance spectrum with single peak.
    # Spectrum calculation is taken from Equation (2) in:
    # Yoon, S.M., Cho, N.J. and Kanazawa, K., 2009. Analyzing spur-distorted 
    # impedance spectra for the QCM. Journal of Sensors, 2009.
    # inputs:
    # Gp = conductance offset
    # Cp = susceptance offset
    # Gmax00 = maximum of conductance peak
    # D00 = dissipation
    # f00 = resonant frequency of peak (peak position) 
    #construct peak
    peak = Gmax00 / (1 + (1j/D00)*((freq/f00)-(f00/freq)))
    #add offsets to spectrum
    Y = Gp + 1j*2*np.pi*freq*Cp + peak
    G = np.real(Y)
    return G

def get_singlebvd_rlc(fit_params0):
    # calculate equivalent circuit parameters from BvD fits
    # from Yoon et al., Analyzing Spur-Distorted Impedance 
    # Spectra for the QCM, Eqn. 3.
        
    #R_fit = []; L_fit = []; C_fit = []
    #deconstruct popt to calculate equivalent circuit parameters
    '''
    for k in range(len(guess0)):
        G0 = popt_real[3*k+2] 
        D0 = popt_real[3*k+3] 
        f0 = popt_real[3*k+4] 
        L0 = 1 / (2 * np.pi * f0 * D0 * G0)
        C0 = 1 / (4 * np.pi**2 * f0**2 * L0)
        R_fit.append(1/G0)
        L_fit.append(L0)
        C_fit.append(C0)
    '''
    #FOR SINGLE BvD PEAK:
    #FIT PARAMS = [Gp, Cp, Gmax00, D00, f00]
    G0 = fit_params0[2]
    f0 = fit_params0[4]
    D0 = fit_params0[3]
    R = 1/G0
    L = 1 / (2 * np.pi * f0 * D0 * G0)
    C = 1 / (4 * np.pi**2 * f0**2 * L)
    return R, L, C, D0

def plot_setup(labels=['X', 'Y'], size=16, setlimits=False, limits=[0,1,0,1]):
    # set axes labels, ranges, and size of labels for a matplotlib plot
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.rcParams.update({'figure.autolayout': True})
    plt.xlabel(str(labels[0]), fontsize=size)
    plt.ylabel(str(labels[1]), fontsize=size)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))


#%% USER INPUTS

filedir = r'C:\Users\a6q\Desktop\AI-controlled experiment\good_data\2019-07-31_14-14_pedotpss'
filename = r'2019-07-31_14-14__qcm_n=1_spectra.csv'

#%% load in data file and drop nan rows
data = pd.read_csv(filedir + '\\' + filename).dropna()
# drop Xs columns
data = data.drop(data.filter(regex='xs').columns, axis=1)
tot_spectra = int(len(list(data))/2)
print('processing %i total spectra' %tot_spectra)

#%% loop over each spectrum

f0_list = []
d_list = []

for col in range(0, tot_spectra*2, 2):

    print('%i / %i' %(int(col/2), tot_spectra))
    freq = data[data.columns[col]].values
    rs = data[data.columns[col+1]].values

    # save resonant frequency
    f0_list.append(freq[np.argmax(rs)])


    # construct guess for peak fit [Gp, Cp, Gmax00, D00, f00]
    guess = [0, 0, np.amax(rs), 1e-3, freq[np.argmax(rs)]]
    try:
        #fit data
        popt, _ = curve_fit(singleBvD_reY, freq, rs,
                            p0=guess, ftol=1e-14, xtol=1e-14,)
        print('fit successful')
    except RuntimeError:
        print('fit failed')
        popt = guess

    fit = singleBvD_reY(freq, *popt)

    # save dissipation
    d_list.append(np.abs(popt[3]))

    # plot spectrum and fit
    plt.scatter(freq, rs, s=4, c='k', marker='o')
    plt.plot(freq, fit, c='r', lw=0.75)
    plot_setup(labels=['Freq.', 'Rs'])
    plt.show()

df_list = f0_list - f0_list[0]
dd_list = 1e6*(d_list- d_list[0])


#%%
plt.plot(df_list)
plot_setup(labels=['Time', '$\Delta$F (Hz)'])
plt.show()

plt.plot(dd_list)
plot_setup(labels=['Time', '$\Delta$D'])
plt.show()

plt.semilogy(dd_list)
plot_setup(labels=['Time', '$\Delta$D'])
plt.show()

     
#%% get delta D from spectra, one harmonic at a time
'''       
key = '9'

#set low and high indices to cut off irrelevant sections of spectra
low_lim, hi_lim = 0, 1500

#frequency values
freq = dic[key][low_lim:hi_lim, 0]

param_dict[key+'_all'] = pd.DataFrame(data = np.zeros((len(dic[key][0])-1, 5)),
                              columns = ['f0', 'd', 'R', 'L', 'C'])


#loop over each spectrum and fit peak to Lorentzian
for i in range(1, len(dic[key][0])):
    print(i)
    
    f0 = param_dict[key]['f0'].iloc[i-1]
    
    g_spec = dic[key][low_lim:hi_lim, i]
    
    #construct guess for peak fit [Gp, Cp, Gmax00, D00, f00]
    guess = [0, 0, np.amax(g_spec), 1e-3, f0]
    
    #fit data
    popt, _ = curve_fit(singleBvD_reY, freq, g_spec,# bounds=(0, np.inf),
                        p0=guess, ftol=1e-14, xtol=1e-14,)
            
    R0, L0, C0, D0 = get_singlebvd_rlc(popt)  
    print('R, L, C, D = ')
    print(R0, L0, C0, D0)
    
    fit = singleBvD_reY(freq, *popt)
    
    plt.plot(freq, g_spec, label='data', c='k')
    plt.plot(freq, fit, label='fit')
    plt.title(i)
    plt.legend()
    plt.show()
    
    
    param_dict[key]['d'].iloc[i-1] = D0

    param_dict[key+'_all']['f0'].iloc[i-1] = f0
    param_dict[key+'_all']['d'].iloc[i-1] = D0
    param_dict[key+'_all']['R'].iloc[i-1] = R0
    param_dict[key+'_all']['L'].iloc[i-1] = L0
    param_dict[key+'_all']['C'].iloc[i-1] = C0
    

plt.plot(param_dict[key]['d'])

'''




#%% extract fit params

'''
dic_params = {'df1':[], 'df3':[], 'df5':[], 'df7':[], 'df9':[], 'df11':[],
              'dd1':[], 'dd3':[], 'dd5':[], 'dd7':[], 'dd9':[], 'dd11':[]}


for i in range(len(good_files)):#[::17]:
    
    data0 = pd.read_csv(good_files[i], skiprows=1).iloc[index1:index2:skip_n,:]
    freq0, g_raw = get_eis_params(data0)
    band = get_band_num(freq0, 10)
    
    
    fit_window = [1000,1400]
    
    freq0 = freq0[fit_window[0]:fit_window[1]]
    g_raw = g_raw[fit_window[0]:fit_window[1]]

    g_raw -= np.min(g_raw)
    
    
    if band == 7:
        print('file %i/%i' %(i+1, len(good_files)+1))

        f0_raw = freq0[np.argmax(g_raw)]
        gmax_raw = np.max(g_raw)
        print('f0 = '+format(f0_raw))
        print('gmax = '+format(gmax_raw))
        
        #fit to Lorentzian BVD equivalent circuit
        #construct guess: [Gp, Cp, Gmax00, D00, f00]
        
        
        guess = [0, 0, gmax_raw, 1e-8, f0_raw]
        
        try:
            popt, _ = curve_fit(singleBvD_reY,
                                freq0,
                                g_raw,
                                # bounds=(0, np.inf),
                                      p0=guess)#, ftol=1e-14, xtol=1e-14,)
            g_fit = multiBvD_reY(freq0, *popt)
            print('fitted peak parameters = '+format(popt))
            print('FIT COMPLETE')
            
            dic_params['df'+str(band)].append(popt[4])
            dic_params['dd'+str(band)].append(popt[3])
            plt.plot(freq0, g_fit, label='fit')
            
        except:
            g_fit = g_raw + 0.0001
            print('FIT FAILED')
        
        plt.plot(freq0, g_raw, label='raw')
        label_axes('Frequency (MHz)', 'Conductance (S)')
        plt.legend()
        plt.show()


        plt.plot(dic_params['df'+str(band)])
        plt.title('dF')
        plt.show()
        
        plt.plot(dic_params['dd'+str(band)])
        plt.title('dd')
        plt.show()
'''
#%%


   
    
'''    
                    
        #save resonant and maximum of spectrum
        f_res0 = freq0[np.argmax(G0)]
        f_res[int(i/band_num)][band_col] = f_res0
        G_max[int(i/band_num)][band_col] = np.max(G0)
        
        #save spectra into matrices
        spec_dict[band_list[band_col]][:, int(i/band_num)+1] = G0
        
        
        
        #plot raw data   
        plt.figure(figsize=(6,5))
        plt.scatter(freq0, G0, label='exp. data', s=10, alpha=0.1, c='k')    
            
        #plot spline fit
        #plt.plot(freq0, G0_spline, c='c', lw=1, label='spline')
            
        #plot sav gol fit
        #plt.plot(freq0, G0_savgol, c='g', lw=1, label='sav-gol')
        
        
    
    
    
    
    
        
        #fit BvD circuit to experimentally measured real conductance (G)######
        if bvd_fit == True: 
            
            
            #try to detect peaks and use them as guesses for peak fitting
            peak_inds, peak_vals = get_peaks(G0_spline, n=1)
    
            #see if enough peaks were found. if not, guess where they are
            if len(peak_inds) < peaks_to_fit:
                peak_inds = np.append(
                        f_res0, [f_res0+50*i for i in range(peaks_to_fit-1)])
                peak_vals = np.append(
                        np.max(G0), np.full(peaks_to_fit-1, np.max(G0)/2))
                
                
            #record number of peaks found
            peak_num_mat[int(i/band_num)][band_col] = len(peak_inds)
        
            #sort by highest to lowest peaks
            peak_order = peak_vals.argsort()[::-1]
            peak_vals, peak_inds = peak_vals[peak_order], peak_inds[peak_order]
            peak_vals = peak_vals[:peaks_to_fit]
            peak_inds = peak_inds[:peaks_to_fit].astype(int)
            
            
            #plot peak positions
            plt.scatter(freq0[peak_inds], peak_vals,
                    label='local max.', s=80, c='y', marker='*')
             
            #plot trajectory of peak position
            #for p in range(peaks_to_fit):
            plt.scatter(peak_ind_mat, peak_val_mat, marker='.',
                        s=5, c='c', alpha=.5)





            
            #construct guess: [Gp, Cp, Gmax00, D00, f00]
            guess = np.array([])
            lowbounds = np.array([])
            highbounds = np.array([])
            #append peak params to guess        
            for k in range(len(peak_inds)):
                guess = np.append(guess, [1e-6, 1e-6, peak_vals[k],
                                          1e-6, freq0[peak_inds[k]]])
                
                #lowbounds = np.append(lowbounds,
                #                      [0, 0, 1e-12, 1e-12, np.min(freq0)])
                #highbounds = np.append(highbounds,
                #                       [np.max(G0), np.max(G0), np.max(G0),
                #                        np.max(freq0)-np.min(freq0),
                #                        np.max(freq0)])
    
    
            #use previous fit params as guess for next ones
            if int(i/band_num) != 0: guess = popt
            
            
            #print('peak fit guess = '+format(guess))
            
            
            
            
            
            
            
            
            
            
            
            
            #fit data
            popt, _ = curve_fit(multiBvD_reY, freq0, G0,# bounds=(0, np.inf),
                                      p0=guess)#, ftol=1e-14, xtol=1e-14,)
            #print('fitted peak parameters = '+format(popt))
            
            fit_params_mat[int(i/band_num)] = np.array(popt)
        
        
           
        
        
        
        
        
        
        
        
        
            
            #plot deconvoluted peaks######################################
            for p in range(0, len(popt), 5):
                peak0 = multiBvD_reY(freq0,
                        popt[p+0], popt[p+1], popt[p+2], popt[p+3], popt[p+4])
                 
                #peak_ind_mat[int(i/band_num),int(p/3)] = popt[p+4]
                #peak_val_mat[int(i/band_num),int(p/3)] = popt[p+2]
                peak_ind_mat.append(popt[p+4])
                peak_val_mat.append(popt[p+2])
                
                
                single_peaks[:,int(p/5)] = peak0
                
                #plot individual fitted peaks
                plt.plot(freq0, peak0, label='peak-'+format(1+int(p/5)),
                         lw=3, alpha=.4)
    
    
    
                #organize RLC fit parameters
                R0, L0, C0, D0 = get_singlebvd_rlc(popt[p:p+5])  
                print('R, L, C, D = ')
                print(R0, L0, C0, D0)
                R_mat[int(i/band_num)][int(p/5)] = np.abs(R0)
                L_mat[int(i/band_num)][int(p/5)] = np.abs(L0)
                C_mat[int(i/band_num)][int(p/5)] = np.abs(C0)
                D_mat[int(i/band_num)][int(p/5)] = np.abs(D0)
    
    
    
    
    
    

            #plot vertical lines showing peak area
            for jj in peak_ind_mat[-peaks_to_fit:]:
                peak_buffer = .008 - 2e-4*jj
                plt.axvline(x=jj,
                            alpha=.2, linestyle=':', c='k')


    
    
            #plot total spectum fit
            G_fit = multiBvD_reY(freq0, *popt)
            plt.plot(freq0, G_fit, label='BvD fit', c='k', lw=0.5)
            
            
            
            
        
        plt.title(format(rh0)+'% RH', fontsize=16)#+', band = '+format(band_list[band_col]), fontsize=18)
        label_axes('Frequency (MHz)', 'Conductance (mS)')
        plt.legend(fontsize=12, ncol=2, loc='upper right')
        plt.tight_layout()
        plt.axis((34.9, 35, -0.05, 2))
        
        
        save_peak_plots = True
        if save_peak_plots == True:
            #save plot as image file            
            save_pic_filename = 'exp_data\\save_figs\\fig'+format(rh0)+'.jpg'
            plt.savefig(save_pic_filename, format='jpg', dpi=150)
        
        plt.show()


'''

#%% normalize f_res matrix by harmonic number
'''
f_res_norm = np.zeros_like(f_res)

for i in range(len(f_res[0])):
    f_res_norm[:,i] = f_res[:,i]/((i*2)+1)  
    plt.plot(f_res_norm[:,i]-f_res_norm[0,i], label=str((i*2)+1))
plt.legend()
label_axes('RH (%)', '$\Delta$f/n (Hz/cm$^{2}$)')
plt.show()

'''





#%% plot spectra and save to txt files
'''
#loop over bands in spectra dictionary
for band in band_list:

    spec_mat = spec_dict[band]
    
    #subtract f0 from F spectra so they are normalized to delta F = 0 @ t = 0
    spec_dict[band][:,0] = spec_dict[band][:,0] - spec_dict[band][
            np.argmax(spec_dict[band][:,1]),0]
        
    #loop over each spectra
    for i in range(1, len(time_table)+1):
                
        plt.plot(spec_mat[:,0], spec_mat[:,i],
                 label=format(int(time_table.pressure.iloc[i-1])))
    label_axes('Frequency (MHz)', 'G conductance (S)')
    #plt.legend()
    plt.show()
    

    #save spectra to file
    save_headers = list((np.array(time_table.pressure
                                          ).astype(int)).astype(str))
    save_headers[0:0] = 'f'
    save_headers = str(' '.join(save_headers)).replace(' ', '\t')
    
    save_spec_filename = 'exp_data\\norm_spec_mat_0'+format(band)+'.txt'
    np.savetxt(save_spec_filename, spec_mat, delimiter='\t',
               header=save_headers, fmt='%.10e', comments='')
    
'''




#%% plot analysis results
'''

#plot change in spectra maximum       
[plt.plot(time_table['pressure'], G_max[:,i] - G_max[0,i],
          label=band_list[i]) for i in range(band_num)]      
label_axes('RH (%)', '$\Delta$ G max ($S$)')
plt.legend(fontsize=14)
plt.show()       




#plot change in resonance frequency
[plt.plot(time_table['pressure'], f_res[:,i] - f_res[0,i],
          label=band_list[i]) for i in range(band_num)]      
label_axes('RH (%)', '$\Delta$ Res. freq. (MHz)')
plt.legend(fontsize=14)
plt.show()




#if BvD fitting was used:
if bvd_fit == True:

    #plot change in R from BVD circuit
    [plt.plot(time_table['pressure'], R_mat[:,i] - R_mat[0,i],
              label='peak-'+format(i+1) for i in range(peaks_to_fit)]      
    label_axes('RH (%)', 'BvD: $\Delta$ R ($\Omega$)')
    plt.legend(fontsize=14)
    plt.show()
    
    #plot change in L from BVD circuit
    [plt.plot(time_table['pressure'], L_mat[:,i] - L_mat[0,i],
              label='peak-'+format(i+1)) for i in range(peaks_to_fit)]      
    label_axes('RH (%)', 'BvD: $\Delta$ L (H)')
    plt.legend(fontsize=14)
    plt.show()
    
    #plot change in C from BVD circuit
    [plt.plot(time_table['pressure'], C_mat[:,i] - C_mat[0,i],
              label='peak-'+format(i+1)) for i in range(peaks_to_fit)]      
    label_axes('RH (%)', 'BvD: $\Delta$ C (F)')
    plt.legend(fontsize=14)
    plt.show()


    #plot change in D from BVD circuit
    [plt.plot(time_table['pressure'], D_mat[:,i] - D_mat[0,i],
              label='peak-'+format(i+1)) for i in range(peaks_to_fit)]      
    label_axes('RH (%)', 'BvD: $\Delta$ D (F)')
    plt.legend(fontsize=14)
    plt.show()





    #plot number of peaks detected
    [plt.plot(time_table['pressure'], peak_num_mat[:,i],
              label=band_list[i]) for i in range(band_num)]      
    label_axes('RH (%)', 'Number of peaks found')
    plt.legend(fontsize=14)
    plt.show()
'''

#%% save figures for creating videos
'''
#band to save
band = 54

#save spectra to file
rh_list = list((np.array(time_table.pressure).astype(int)))

#loop over RH
for i in range(0, len(rh_list)):
       
    plt.plot(spec_dict[band][:,0]*1000, spec_dict[band][:,i+1],c='navy',lw=1)
    label_axes('$\Delta$F (kHz/cm$^2$)', 'Norm. conductance')
    
    plt.title(format(rh_list[i])+'% RH', fontsize=18)

    plt.axis((-100, 100, 0, 1))
    plt.tight_layout()
    
    save_pic_filename = 'exp_data\\save_figs\\fig'+format(rh_list[i])+'.jpg'
    plt.savefig(save_pic_filename, format='jpg', dpi=150)

    plt.show()
'''





#plt.savefig('sample_save_spyder_fig.jpg', format='jpg', dpi=1000)


#%% plot heat maps of QCM response

'''   
#loop over each band in band dictionary
for key in spec_dict:

    #reshape spec_mat into columns
    Xf, Yf, Zf = np.array([]), np.array([]), np.array([])
    
    #loop over each pressure
    for i in range(len(time_table)):
        
        #set X values = pressure 
        Xf = np.append(Xf, np.repeat(
                np.array(time_table['pressure'].iloc[i]), new_spec_len))
        #set Y values = frequency
        Yf = np.append(Yf, spec_dict[key][:,0])
        #set Z values = G
        Zf = np.append(Zf, spec_dict[key][:,i+1])
    
    
    
    # create x-y points to be used in heatmap
    xf = np.linspace(Xf.min(),Xf.max(),100)
    yf = np.linspace(Yf.min(),Yf.max(),100)
    # Z is a matrix of x-y values
    zf = griddata((Xf, Yf), Zf, (xf[None,:], yf[:,None]), method='cubic')
    # Create the contour plot
    CSf = plt.contourf(xf, yf, zf, 100, cmap=plt.cm.rainbow, 
                       vmax=np.nanmax(Zf), vmin=np.nanmin(Zf))
    plt.colorbar()
    label_axes('RH (%)', 'F$_0$ (Hz/cm$^2$)')
    plt.show()
    
'''



#%% manual peak construction
        
      
'''       
guess = [1.00e-06, 1.00e-06, 1e3*1.00e+00, 5, 4.11e+02]

G_fit = multiBvD_reY(freq0*1e6, *guess)

plt.scatter(freq0*1e6, G0*1e3)
plt.plot(freq0*1e6, G_fit*1e3)   
        
'''       

'''
for p in range(peaks_to_fit):
    plt.plot(peak_ind_mat[:, p], peak_val_mat[:,p])
plt.axis((24.9, np.max(freq0), -0.02, 1.02))
plt.show()
'''






