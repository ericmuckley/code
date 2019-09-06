'''
This module provides methods and data structures for comparing the
performance of different machine learning models, including
plotting, spliting train/test input data sets, resampling arrays,
comparing different fits, file import/export, and multipeak fitting.
'''

import matplotlib.pyplot as plt

def plt_setup(
    labels=['X', 'Y'], title=None, size=16, setlimits=False, limits=[0,1,0,1]):
    '''
    This can be called before displaying a matplotlib figure to set
    axes labels, axes ranges, and set the font size of plot labels.
    Function should be called between plt.plot() and plt.show() commands.
    '''
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.rcParams.update({'figure.autolayout': True})
    plt.xlabel(str(labels[0]), fontsize=size)
    plt.ylabel(str(labels[1]), fontsize=size)
    if title is not None:
        plt.title(title, fontsize=size)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))


import bokeh
from bokeh.plotting import figure
from bokeh.io import output_notebook, show

def setup_bokeh_fig(title='Title', xlabel='X', ylabel='Y', fontsize=12):
    """
    Setup interactive Bokeh plot.
    After calling this function and adding series to the plot, use:
    output_notebook()
    show(p)
    """
    # create the figure
    p = figure(width=600, height=400,
               title=title,
               x_axis_label=xlabel,
               y_axis_label=ylabel,
               tools="pan,wheel_zoom,box_zoom,reset")
    # adjust properties of the figure to make it look good
    p.toolbar.logo = None
    fs = str(int(fontsize))+'pt'
    p.title.text_font_size = fs
    p.xaxis.axis_label_text_font_size = fs
    p.yaxis.axis_label_text_font_size = fs
    p.xaxis.major_label_text_font_size = fs
    p.yaxis.major_label_text_font_size = fs
    p.xaxis.axis_label_text_font_style = "normal"
    p.yaxis.axis_label_text_font_style = "normal"
    return p


import itertools
from bokeh.palettes import Dark2_8 as palette

def plot_line_series(df, title='Title', xlabel='X', ylabel='Y', fontsize=12):
    """
    Create interactive Bokeh line plot with multiple lines.
    Input df should be Pandas Dataframe, where the first column
    is x-values and following columns are y-values.
    This function calls 'setup_bokeh_fig' to place labels and formatting.
    """
    p = setup_bokeh_fig(
        title=title, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize)
    colors = itertools.cycle(palette)
    # loop over each column in the data and add to plot
    for col, color in zip(range(1, len(df.columns)), colors):
        p.line(df[df.columns[0]],
               df[df.columns[col]],
               line_color=color,
               line_width=4, legend=df.columns[col], alpha=0.6,
               muted_color=color, muted_alpha=0.2)
    p.legend.click_policy = "mute"
    # show the plot after this function using show(p)
    output_notebook()
    return p


from scipy.interpolate import splrep
from scipy.interpolate import splev


def arr_resample(arr, new_len=100, new_xlims=None, vec_scale='lin', k=3, s=0):
    '''
    Resamples (stetches/compresses) a 2D array by using a spline fit.
    Array should be shape [[x1, y1, ...ym], ...[xn, yn, ...yn]] where the
    # first column in array is x-values and following next columns are
    y values. If no x values exist, insert column np.arange(len(arr))
    as x values.
    Accepts linear or log x-values, and new x_limits.
    k and s are degree and smoothing factor of the interpolation spline.
    '''
    # first, check whether array should be resampled using
    # a linear or log scale:
    if vec_scale == 'lin':
        new_scale = np.linspace
    if vec_scale == 'log':
        new_scale = np.geomspace
    # get new x-limits for the resampled array
    if new_xlims is None:
        new_x1, new_x2 = arr[0, 0], arr[-1, 0]
    else:
        new_x1, new_x2 = new_xlims[0], new_xlims[1]
    # create new x values
    arrx = new_scale(new_x1, new_x2, new_len)
    # create new empty array to hold resampled values
    stretched_array = np.zeros((new_len, len(arr[0])))
    stretched_array[:, 0] = arrx 
    # for each y-column, calculate parameters of degree-3 spline fit
    for col in range(1, len(arr[0])):
        spline_params = splrep(arr[:, 0], arr[:, col], k=int(k), s=s)
        # calculate spline at new x values
        arry = splev(arrx, spline_params)
        # populate stretched data into resampled array
        stretched_array[:, col] = arry
    return stretched_array


def normalize_df(df, norm_x=False):
    '''
    Normalize a Pandas Dataframe by looping over each column and setting the
    range of the column from 0 - 1. This function does NOT normalize the first
    column by default, which is assumed to be the independent variable.
    Set argument norm_x=True to normalize the first column as well.
    '''
    start_index = 0 if norm_x else 1
    # loop over each column
    for i in range(start_index, len(list(df))):
        # subtract the minimum value
        df.iloc[:, i] -= df.iloc[:, i].min()
        # divide the maximum value
        df.iloc[:, i] /= df.iloc[:, i].max()
    return df


def format_for_ml(df, target_feature, train_samples,
                  drop_features=None, scaler=StandardScaler()):
    '''
    Splits a Pandas DataFrame (df) into different arrays for 
    use in ML models. The df is split into input and output (target)
    feature by column name. Columns which are not target_feature
    are used as input features. 

    The df is split into training and testing arrays according
    to the list of indices (rows) designated by train_samples.
    The rows which are not used for training are used as test_samples.
    Features can be dropped by using a list of column names in the
    drop_features agument.
    
    Use a scaler to create scaled arrays for use in the ML model.
    The output is a dictionary:
    in_train: input features for training
    out_train: target feature for training
    in_test: input features for testing
    out_test: target feature for testing
    in_train_s: scaled input features for training
    out_train_s: scaled target feature for training
    in_test_s: scaled input features for testing
    out_test_s: scaled target feature for testing
    '''
    df2 = df.copy()
    # drop unwanted features
    if drop_features is not None:
        df2.drop(drop_features, inplace=True, axis=1)

    # for input features, use all columns that are not used as target_feature
    input_features = [col for col in list(df2) if col != target_feature]
    # for test samples, use all samples that are not used as training samples
    test_samples = np.setdiff1d(df2.index.values, train_samples)

    # create datasets
    in_train = df2[input_features].iloc[train_samples]
    out_train = df2[target_feature].iloc[train_samples]
    in_test = df2[input_features].iloc[test_samples]
    out_test = df2[target_feature].iloc[test_samples]

    # fit the scaler on all the input data
    scaler.fit(df2[input_features].values)
    
    # standardize the input datasets using the scaler
    in_train_s = scaler.transform(in_train.values)
    in_test_s = scaler.transform(in_test.values)

    # store datasets in a dictionary
    ds = {'in_train': in_train,
          'out_train': out_train,
          'in_test': in_test,
          'out_test': out_test,
          'in_train_s': in_train_s,
          'in_test_s': in_test_s,
          'scaler': scaler,
          'target_feature': target_feature}
    return ds


def examine_ml_datasets(ds):
    # examine the datasets produced by the "format_for_ml" function
    # loop over each dataset in the dictionary
    for d in ds:
        if isinstance(ds[d], pd.DataFrame):
            print('%s: shape=%s, columns=%s' %(
                str(d), str(ds[d].shape), str(list(ds[d]))))
        if isinstance(ds[d], np.ndarray):
            print('%s: shape=%s' %(str(d), str(ds[d].shape)))
        if isinstance(ds[d], str):
            print('%s: %s' %(d, ds[d]))


import keras
from keras import optimizers
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


class GetWeights(Callback):
    '''
    Custom Keras callback which collects values of weights and biases
    after each epoch of training the model. Should be used like this:
    gw = GetWeights()
    model.fit(callbacks=[gw])
    wd = gw.weight_dict
    '''
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}
    def on_epoch_end(self, epoch, logs=None):
        # this function runs at the end of each epoch
        # loop over each layer and get weights and biases
        for layer_i in range(len(self.model.layers)):
            w = self.model.layers[layer_i].get_weights()[0]
            b = self.model.layers[layer_i].get_weights()[1]
            # save all weights and biases inside a dictionary
            if epoch == 0:
                # create array to hold weights and biases
                self.weight_dict['w_'+str(layer_i+1)] = w
                self.weight_dict['b_'+str(layer_i+1)] = b
            else:
                # append new weights to previously-created weights array
                self.weight_dict['w_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['w_'+str(layer_i+1)], w))
                # append new weights to previously-created weights array
                self.weight_dict['b_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['b_'+str(layer_i+1)], b))



import matplotlib.gridspec as gridspec
from matplotlib import cm

wd = gw.weight_dict

def plot_weights(wd):
    '''
    Takes a dictionary of weights and biases from the Keras callback
    'GetWeights' and plots them as heatmaps for each training epoch.
    '''
    # loop over each epoch in the training session
    for epoch in range(0, 3, 1):
        # layout the figure based on the number of layers
        fig = plt.figure(figsize=(15, 6))
        gs_stretch = 6
        gs = gridspec.GridSpec(6, int(len(wd)/2)*gs_stretch, figure=fig)
        gs.update(
            left=0.1, right=0.9, top=0.85, bottom=0.03, wspace=0.01, hspace=2)

        # loop over each array of weights and biases
        for key_i, key in enumerate(wd):

            # check if key is a weights key
            if 'w' in key:
                # row and column to place plot
                row = gs.get_geometry()[0]-1
                col = gs_stretch*int(key_i/2)+int(key_i/2)
                # normal layer
                if key != list(wd.keys())[-2]:
                    plt.subplot(gs[:row, col:col+gs_stretch]).axis('off')
                    plt.gca().set_title(
                        'L-'+str(1+int(key_i/2))+' weights', fontsize=18)
                # output layer
                else:
                    plt.subplot(gs[:row, col]).axis('off')
                    plt.gca().set_title('Output\nweights', fontsize=18)

            # check if key is a biases key
            if 'b' in key:
                # normal layer
                if key != list(wd.keys())[-1]:
                    plt.subplot(gs[row, col:col+gs_stretch]).axis('off') 
                    plt.gca().set_title(
                        'L-'+str(1+int(key_i/2))+' biases', fontsize=18) 
                # output layer
                else:
                    plt.subplot(gs[row, col]).axis('off')
                    plt.gca().set_title('Output\nbiases', fontsize=18)

            # generate heatmap of weight/bias values
            plt.imshow(wd[key][:,:,epoch],
                    aspect='auto',
                    cmap=plt.get_cmap('coolwarm'),
                    interpolation='nearest')

        fig.suptitle('Epoch '+str(epoch), fontsize=22)
        plt.axis('off')
        plt.show()

        # save the image to file
        save_heatmaps = False
        if save_heatmaps:
            fig_filename = 'epoch_'+str(epoch).zfill(5)+'.png'
            fig.savefig(fig_filename, dpi=200)
            files.download(fig_filename)


def multigauss(signal, *params):
    '''
    Function for fitting a signal with multiple gaussian peaks.
    Signal is the y value, and *params is a list of fitting parameters
    of the form [center, amplitude, width, center, amplitude, width...]
    '''
    y = np.zeros_like(signal)
    # for each gauss peak get the center, amplitude, and width
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((signal - ctr)/wid)**2)
    # return the sum of all the peaks   
    return y


def get_spline(x, y, k=3, s=10):
    '''
    Get the spline fit for a curve using X and Y data.
    k is the order of the spline fit, s is the smoothing parameter.
    '''
    from scipy.interpolate import splrep, splev
    # calculate spline parameters
    spline_params = splrep(x, y, k=k, s=s)
    # create the spline curve
    spline = splev(x, spline_params)
    return spline


def detect_peaks(x, y, window=21, prominence=0.01):
    '''
    Get peaks from a spectrum of x and y values.
    The prominence_visible argument sets the prominence threshold for what
    constitutes a visible peak.
    The prominence_hidden argument sets the prominence threshold for what
    constitutes a hidden peak.
    The hidden argument specifies whether to search for hidden peaks,
    i.e. features which do not create visible peaks in the spectrum,
    but are buried or convoluted by other peaks.
    The window argument specific window length for smoothing the spectrum.
    This must be an odd integer.
    Returns a 2D array of x and y columns which correspond to peak positoins
    and peak heights.
    '''
    from scipy.signal import savgol_filter, find_peaks
    # make sure inputs are array-like
    x, y_norm = np.array(x), np.array(y)
    # normalize the y data
    y_norm -= np.min(y)
    y_norm /= np.max(y)
    # if window length is an even integer, make it odd
    if window % 2 == 0:
        window += 1
    # get 1st derivative of signal
    deriv1 = savgol_filter(y_norm, window_length=window, polyorder=3, deriv=1)
    # get 2nd derivative of signal
    deriv2 = savgol_filter(y_norm, window_length=window, polyorder=3, deriv=2)
    # find indices where 1st derivative is 0
    zero_indices = np.where(np.diff(np.signbit(deriv1)))[0]
    # select only idices where 2nd derivative < 0
    peak_ind = [i for i in zero_indices if deriv2[i] < 0]
    # find troughs in 2nd derivative, which may correspond to hidden peaks
    hidden_ind = find_peaks(-deriv2, prominence=prominence)[0]
    peak_ind.extend(hidden_ind)
    peaks = np.stack((x[peak_ind], y[peak_ind]), axis=1)
    return peaks



import pickle

def import_pickle(filename):
    '''
    Opens serialized Python pickle file as a dictionary.
    Filename should be something like 'saved_data.pkl'.
    '''
    with open(filename, 'rb') as handle:
        dic = pickle.load(handle)
    return dic

def export_pickle(filename, dic):
    '''
    Serializes Python dictionary as a pickle file.
    Filename should be something like 'saved_data.pkl'.
    '''
    with open(clean_data_filename, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)



import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from scipy.interpolate import splrep
from scipy.interpolate import splev

import sklearn.linear_model as lm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR

def compare_fits(
    x, y, x_new=None, fits=['poly1', 'poly2', 'poly3', 'spline'], s=5):
    '''
    Compares least squares fits to x and y ordered pairs.
    Performs fits at x_new points. The fits argument is a
    list of strings which indicate which fits to compare.
    fits = [
        'poly1' = degree 1 polynomial (linear) fit 
        'poly2' = degree 2 polynomial (quadratic) fit
        'poly3' = degree 3 polynomial fit
        'spline' = B-spline fit
            ]
    s = smoothing factor for the spline fit
    '''
    # if new x values are not passed, use original x values
    if x_new is None:
        x_new = x
    # create dictionary to hold fits
    fit_dic = {}
    # perform fits
    if 'poly1' in fits:
        poly1coefs = poly.polyfit(x, y, 1)
        poly1fit = poly.polyval(x_new, poly1coefs)
        fit_dic['poly1'] = poly1fit
    if 'poly2' in fits:
        poly2coefs = poly.polyfit(x, y, 2)
        poly2fit = poly.polyval(x_new, poly2coefs)
        fit_dic['poly2'] = poly2fit
    if 'poly3' in fits:
        poly3coefs = poly.polyfit(x, y, 3)
        poly3fit = poly.polyval(x_new, poly3coefs)
        fit_dic['poly3'] = poly3fit
    if 'spline' in fits:
        spline_params = splrep(x, y, s=s, k=3)
        splinefit = splev(x_new, spline_params)
        fit_dic['spline'] = splinefit
    return fit_dic


modeldict = {
    'ardregression': lm.ARDRegression(),
    'bayesianridge': lm.BayesianRidge(),
    'elasticnet': lm.ElasticNet(),
    'elasticnetcv': lm.ElasticNetCV(),
    'huberregression': lm.HuberRegressor(),
    'lars': lm.Lars(),
    'larscv': lm.LarsCV(),
    'lasso': lm.Lasso(),
    'lassocv': lm.LassoCV(),
    'lassolars': lm.LassoLars(),
    'lassolarscv': lm.LassoLarsCV(),
    'lassolarsic': lm.LassoLarsIC(),
    'linearregression': lm.LinearRegression(),
    'orthogonalmatchingpursuit': lm.OrthogonalMatchingPursuit(),
    'orthogonalmatchingpursuitcv': lm.OrthogonalMatchingPursuitCV(),
    'passiveagressiveregressor': lm.PassiveAggressiveRegressor(),
    'ridge': lm.Ridge(),
    'ridgecv': lm.RidgeCV(),
    'sgdregressor': lm.SGDRegressor(),
    'theilsenregressor': lm.TheilSenRegressor(),
    'decisiontreeregressor': DecisionTreeRegressor(),
    'randomforestregressor': RandomForestRegressor(),
    'adaboostregressor': AdaBoostRegressor(),
    'baggingregressor': BaggingRegressor(),
    'extratreeregressor': ExtraTreeRegressor(),
    'linearsvr': LinearSVR(),
    'nusvr': NuSVR(),
    'svr': SVR(),
    }

