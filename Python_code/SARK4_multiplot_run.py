
ls = 16
plt.rcParams['xtick.labelsize'], plt.rcParams['ytick.labelsize'] = ls, ls

s = 10

plt.rcParams['figure.figsize'] = (6,6)

import matplotlib.gridspec as gridspec

for i in range(43, len(rh_list)):
    

    
    gs1 = gridspec.GridSpec(3,1)
    gs2 = gridspec.GridSpec(3,1)
    gs3 = gridspec.GridSpec(3,1)
    gs1.update(left=0.05, right=0.4)#, wspace=0.05)
    gs2.update(left=0.48, right=0.7)#, wspace=0.05)
    gs3.update(left=0.78, right=0.98)

    
    ax_spec = plt.subplot(gs1[:])
    ax_f = plt.subplot(gs2[0:-1])
    ax_D = plt.subplot(gs2[2])
    ax_R = plt.subplot(gs3[0])
    ax_L = plt.subplot(gs3[1])
    ax_C = plt.subplot(gs3[2])
    
    share_axes1 = [ax_f, ax_D]
    share_axes2 = [ax_R, ax_L, ax_C]

    plt.setp([a.get_xticklabels() for a in share_axes1[:-1]], visible=False)
    plt.setp([a.get_xticklabels() for a in share_axes2[:-1]], visible=False)
    plt.subplots_adjust(hspace=0, bottom=.15, top=0.95, right=0.3, left=.1)
    
    '''
    #WORKS:
    gs = gridspec.GridSpec(4,2)
    ax0 = plt.subplot(gs[:,0])
    ax1 = plt.subplot(gs[0,1])
    ax2 = plt.subplot(gs[1,1])
    ax3 = plt.subplot(gs[2,1])
    ax4 = plt.subplot(gs[3,1])
    '''


    
    #set up multi-plot figure
    #fig, (ax01, ax02, ax03, ax04,
    #      ax1, ax2, ax3, ax4) = plt.subplots(4, 2, sharex=True, figsize=(6,10))
    

    #plt.legend(fontsize=14)
    
    for j in range(0, len(R_mat[0])):

        rh_list2 = rh_list[:i+1]

        R_list = (R_mat[:i+1, j] - R_mat[0, j])
        C_list = (C_mat[:i+1, j] - C_mat[0, j])*1e9
        L_list = L_mat[:i+1, j] - L_mat[0, j]
        D_list = (D_mat[:i+1, j] - D_mat[0, j])*1e6
        
        #plot R
        ax_R.scatter(rh_list2, R_list, s=s, label='peak-'+format(j+1))
        ax_R.plot(rh_list2, R_list)
        #plot L
        ax_L.scatter(rh_list2, L_list, s=s, label='peak-'+format(j+1))
        ax_L.plot(rh_list2, L_list)
        #plot C
        ax_C.scatter(rh_list2, C_list, s=s, label='peak-'+format(j+1))
        ax_C.plot(rh_list2, C_list)
        #plot D
        ax_D.scatter(rh_list2, D_list, s=s, label='peak-'+format(j+1))
        ax_D.plot(rh_list2, D_list)
        
    ax_D.set_xlim(0,100) 
    ax_C.set_xlim(0,100) 
    #ax1.set_ylim(-0.3,7) 
    #ax2.set_ylim(-250,99)
    #ax3.set_ylim(-70,25)
    #ax4.set_ylim(-3,65)
    ax_spec.set_ylabel('Conductance (mS)', fontsize=ls)  
    ax_f.set_ylabel('$\Delta$f (Hz/cm$^{2}$)', fontsize=ls)  
    ax_R.set_ylabel('$\Delta$R ($\Omega$)', fontsize=ls)    
    ax_L.set_ylabel('$\Delta$L (H)', fontsize=ls)
    ax_C.set_ylabel('$\Delta$C (nF)', fontsize=ls)   
    ax_D.set_ylabel('$\Delta$D (x10$^{-6})$', fontsize=ls)
    ax_D.set_xlabel('Relative Humidity (%)', fontsize=ls) 
    ax_C.set_xlabel('Relative Humidity (%)', fontsize=ls)
    ax_spec.set_xlabel('Frequency (MHz)', fontsize=ls)
    #ax1.legend(loc='upper left', fontsize=14)  
    #plt.legend(fontsize=14, loc='upper left')
    #ax2.legend(loc='upper right')
    #ax3.legend(loc='upper right')
    #ax4.legend(loc='upper left', fontsize=14)
    #plt.legend(fontsize=14, loc='upper left')
    
    #save plot as image file          
    #plt.tight_layout()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 6)
    save_pic_filename = 'exp_data\\save_figs2\\fig'+format(rh_list[i])+'.jpg'
    plt.savefig(save_pic_filename, format='jpg', dpi=150)

    plt.show()
    
#close figure from memory
plt.close(figure)
#close all figures from memory
plt.close("all")