%Inputs for peakfit are:
% 2D data, center of data, data window width, # of peaks,
% 1 = Gaussian fitting, iterations, vector of guesses and widths [x1 w1 x2
% w2 ...], 
%

%make up example data to fit
x = 0:.1:20;
data = sin(x)+1;

%run the peak fitting
[FitResults] = multiplepeakfit_GAUSS(data, 0, 0, 8, 1, 5, 0);

meanheight = mean(FitResults(:,3))
meanwidth = mean(FitResults(:,4))

hold off



% making multiple trial fits with sightly different starting
% values (whose variability is controled by the 14th input argument) and
% taking the one with the lowest mean fit error (example 6). It can
% estimate the standard deviation of peak parameters from a single signal
% using the bootstrap method (example 10).
%
% For more details, see
% http://terpconnect.umd.edu/~toh/spectrum/CurveFittingC.html and
% http://terpconnect.umd.edu/~toh/spectrum/InteractivePeakFitter.htm
%
% peakfit(signal);       
% Performs an iterative least-squares fit of a single Gaussian peak to the
% data matrix "signal", which has x values in column 1 and Y values in
% column 2 (e.g. [x y])
%
% peakfit(signal,center,window);
% Fits a single Gaussian peak to a portion of the matrix "signal". The
% portion is centered on the x-value "center" and has width "window" (in x
% units).
% 
% peakfit(signal,center,window,NumPeaks);
% "NumPeaks" = number of peaks in the model (default is 1 if not
% specified).
% 
% peakfit(signal,center,window,NumPeaks,peakshape); 
% "peakshape" specifies the peak shape of the model: (1=Gaussian (default),
% 2=Lorentzian, 3=logistic distribution, 4=Pearson, 5=exponentionally
% broadened Gaussian; 6=equal-width Gaussians; 7=Equal-width Lorentzians;
% 8=exponentionally broadened equal-width Gaussian, 9=exponential pulse,
% 10=up-sigmoid (logistic function), 11=Fixed-width Gaussian,
% 12=Fixed-width Lorentzian; 13=Gaussian/ Lorentzian blend; 14=Bifurcated
% Gaussian, 15=Breit-Wigner-Fano, 16=Fixed-position Gaussians;
% 17=Fixed-position Lorentzians; 18=exponentionally broadened Lorentzian;
% 19=alpha function; 20=Voigt profile; 21=triangular; 22=multiple shapes;
% 23=down-sigmoid; 25=lognormal; 26=slope; 27=Gaussian first derivative; 
% 28=polynomial; 29=piecewise linear; 30=variable-alpha Voigt; 31=variable
% time constant ExpGaussian; 32=variable Pearson; 33=variable
% Gaussian/Lorentzian blend
%
% peakfit(signal,center,window,NumPeaks,peakshape,extra) 
% 'extra' specifies the value of 'extra', used only in the Voigt, Pearson,
% exponentionally broadened Gaussian, Gaussian/Lorentzian blend, and
% bifurcated Gaussian and Lorentzian shapes to fine-tune the peak shape.
% 
% peakfit(signal,center,window,NumPeaks,peakshape,extra,NumTrials);
% Performs "NumTrials" trial fits and selects the best one (with lowest
% fitting error). NumTrials can be any positive integer (default is 1).
%
% peakfit(signal,center,window,NumPeaks,peakshape,extra,NumTrials,start)
% Specifies the first guesses vector "firstguess" for the peak positions
% and widths. Must be expressed as a vector , in square brackets, e.g.
% start=[position1 width1 position2 width2 ...]
% 
% peakfit(signal,center,window,NumPeaks,peakshape,extra,NumTrials,start,autozero) 
% 'autozero' sets the baseline correction mode:
% autozero=0 (default) does not subtract baseline from data segment; 
% autozero=1 interpolates a linear baseline from the edges of the data 
%            segment and subtracts it from the signal (assumes that the 
%            peak returns to the baseline at the edges of the signal); 
% autozero=2 is like mode 1 except that it computes a quadratic curved baseline; 
% autozero=3 compensates for a flat baseline without reference to the 
%            signal itself (best if the peak does not return to the
%            baseline at the edges of the signal).
%
% peakfit(signal,center,window,NumPeaks,peakshape,extra,NumTrials,start,...
% autozero,fixedparameters)
% 'fixedparameters' specifies fixed values for widths (shapes 10, 11) or
% positions (shapes 16, 17)
% 
% peakfit(signal,center,window,NumPeaks,peakshape,extra,NumTrials,start,...
% autozero,fixedparameters,plots)
% 'plots' controls graphic plotting: 0=no plot; 1=plots draw as usual (default)
% 
% peakfit(signal,center,window,NumPeaks,peakshape,extra,NumTrials,start,...
% autozero,fixedparameters,plots,bipolar)
% 'bipolar' = 0 constrain peaks heights to be positions; 'bipolar' = 1
% allows positive ands negative peak heights.
%
% peakfit(signal,center,window,NumPeaks,peakshape,extra,NumTrials,start,...
% autozero,fixedparameters,plots,bipolar,minwidth)
% 'minwidth' sets the minmimum allowed peak width. The default if not
% specified is equal to the x-axis interval. Must be a vector of minimum
% widths, one value for each peak, if the multiple peak shape is chosen, 
% as in example 17 and 18.
%
% peakfit(signal,center,window,NumPeaks,peakshape,extra,NumTrials,start,...
% autozero,fixedparameters,plots,bipolar,minwidth)
%  'DELTA' (14th input argument) controls the restart variance when
%  NumTrials>1. Default value is 1.0. Larger values give more variance.
%  Version 5.8 and later only. 
% [FitResults,FitError]=peakfit(signal,center,window...) Returns the
% FitResults vector in the order peak number, peak position, peak height,
% peak width, and peak area), and the FitError (the percent RMS
% difference between the data and the model in the selected segment of that
% data) of the best fit.
%
% [FitResults,LowestError,BestStart,xi,yi,BootResults]=peakfit(signal,...)
% Prints out parameter error estimates for each peak (bootstrap method).
%
% Optional output parameters 
% 1. FitResults: a table of model peak parameters, one row for each peak,
%    listing Peak number, Peak position, Height, Width, and Peak area.
% 2. GOF: Goodness of Fit, a vector containing the rms fitting error of the
%    best trial fit and the R-squared (coefficient of determination).
% 3. Baseline, the polynomial coefficients of  the baseline in linear
%    and quadratic baseline modes (1 and 2) or the value of the constant
%    baseline in flat baseline mode.
% 3. coeff: Coefficients for the polynomial fit (shape 28 only; for other 
%    shapes, coeff=0)
% 5. residual: the difference between the data and the best fit.
% 6. xi: vector containing 600 interploated x-values for the model peaks. 
% 7. yi: matrix containing the y values of each model peak at each xi. 
%    Type plot(xi,yi(1,:)) to plot peak 1 or plot(xi,yi) to plot all peaks
% 8. BootResults: a table of bootstrap precision results for a each peak
%    and peak parameter.
%  
% Example 1: 
% >> x=[0:.1:10]';y=exp(-(x-5).^2);peakfit([x y])
% Fits exp(-x)^2 with a single Gaussian peak model.
%
%        Peak number  Peak position   Height     Width      Peak area
%             1            5            1        1.665       1.7725
%
% >> y=[0 1 2 4 6 7 6 4 2 1 0 ];x=1:length(y);
% >> peakfit([x;y],length(y)/2,length(y),0,0,0,0,0,0)
% Fits small set of manually entered y data to a single Gaussian peak model.
%
% Example 2:
% x=[0:.01:10];y=exp(-(x-5).^2)+randn(size(x));peakfit([x;y])
% Measurement of very noisy peak with signal-to-noise ratio = 1.
% ans =
%             1         5.0279       0.9272      1.7948      1.7716
%
% Example 3:
% x=[0:.1:10];y=exp(-(x-5).^2)+.5*exp(-(x-3).^2)+.1*randn(size(x));
% peakfit([x' y'],0,0,2)
% Fits a noisy two-peak signal with a double Gaussian model (NumPeaks=2).
% ans =
%             1       3.0001      0.49489        1.642      0.86504
%             2       4.9927       1.0016       1.6597       1.7696
%
% Example 4:
% >> x=1:100;y=ones(size(x))./(1+(x-50).^2);peakfit(y,0,0,1,2)
% Fit Lorentzian (peakshape=2) located at x=50, height=1, width=2.
% ans =
%            1           50      0.99974       1.9971       3.1079
%
% Example 5: 
% >> x=[0:.005:1];y=humps(x);peakfit([x' y'],.3,.7,1,4,3);
% Fits a portion of the humps function, 0.7 units wide and centered on 
% x=0.3, with a single (NumPeaks=1) Pearson function (peakshape=4) with
% extra=3 (controls shape of Pearson function).
%
% Example 6: 
% >> x=[0:.005:1];y=(humps(x)+humps(x-.13)).^3;smatrix=[x' y'];
% >> [FitResults,FitError]=peakfit(smatrix,.4,.7,2,1,0,10)
% Creates a data matrix 'smatrix', fits a portion to a two-peak Gaussian
% model, takes the best of 10 trials.  Returns FitResults and FitError.
% FitResults =
%              1      0.31056  2.0125e+006      0.11057  2.3689e+005
%              2      0.41529  2.2403e+006      0.12033  2.8696e+005
% FitError =
%         1.1899
%
% Example 7:
% >> peakfit([x' y'],.4,.7,2,1,0,10,[.3 .1 .5 .1]);
% As above, but specifies the first-guess position and width of the two
% peaks, in the order [position1 width1 position2 width2]
%
% Example 8: (Version 4 only)
% Demonstration of the four autozero modes, for a single Gaussian on flat
%  baseline, with position=10, height=1, and width=1.66. Autozero mode
%  is specified by the 9th input argument (0,1,2, or 3).
% >> x=8:.05:12;y=1+exp(-(x-10).^2);
% >> [FitResults,FitError]=peakfit([x;y],0,0,1,1,0,1,0,0)
% Autozero=0 means to ignore the baseline (default mode if not specified)
% FitResults =
%             1           10       1.8561        3.612       5.7641
% FitError =
%         5.387
% >> [FitResults,FitError]=peakfit([x;y],0,0,1,1,0,1,0,1)
% Autozero=1 subtracts linear baseline from edge to edge.
% Does not work well because signal does not return to baseline at edges.
% FitResults =
%             1       9.9984      0.96153        1.559       1.5916
% FitError =
%        1.9801
% >> [FitResults,FitError]=peakfit([x;y],0,0,1,1,0,1,0,2)
% Autozero=1 subtracts quadratic baseline from edge to edge.
% Does not work well because signal does not return to baseline at edges.
% FitResults =
%             1       9.9996      0.81749       1.4384       1.2503
% FitError =
%        1.8204
% Autozero=3: Flat baseline mode, measures baseline by regression
% >> [FitResults,Baseline,FitError]=peakfit([x;y],0,0,1,1,0,1,0,3)
% FitResults =
%             1           10       1.0001       1.6653       1.7645
% Baseline =
%     0.0037056
% FitError =
%       0.99985
%
% Example 9:
% x=[0:.1:10];y=exp(-(x-5).^2)+.5*exp(-(x-3).^2)+.1*randn(size(x));
% [FitResults,FitError]=peakfit([x' y'],0,0,2,11,0,0,0,0,1.666)
% Same as example 3, fit with fixed-width Gaussian (shape 11), width=1.666
% 
% Example 10: (Version 3 or later; Prints out parameter error estimates)
% x=0:.05:9;y=exp(-(x-5).^2)+.5*exp(-(x-3).^2)+.01*randn(1,length(x));
% [FitResults,LowestError,residuals,xi,yi,BootstrapErrors]=peakfit([x;y],0,0,2,6,0,1,0,0,0);
%
% Example 11: (Version 3.2 or later)
% x=[0:.005:1];y=humps(x);[FitResults,FitError]=peakfit([x' y'],0.54,0.93,2,13,15,10,0,0,0) 
%
% FitResults =
%             1      0.30078       190.41      0.19131       23.064
%             2      0.89788       39.552      0.33448       6.1999
% FitError = 
%       0.34502
% Fits both peaks of the Humps function with a Gaussian/Lorentzian blend
% (shape 13) that is 15% Gaussian (Extra=15).
% 
% Example 12:  (Version 3.2 or later)
% >> x=[0:.1:10];y=exp(-(x-4).^2)+.5*exp(-(x-5).^2)+.01*randn(size(x));
% >> [FitResults,FitError]=peakfit([x' y'],0,0,1,14,45,10,0,0,0) 
% FitResults =
%             1       4.2028       1.2315        4.077       2.6723
% FitError =
%       0.84461
% Fit a slightly asymmetrical peak with a bifurcated Gaussian (shape 14)
% 
% Example 13:  (Version 3.3 or later)
% >> x=[0:.1:10]';y=exp(-(x-5).^2);peakfit([x y],0,0,1,1,0,0,0,0,0,0)
% Example 1 without plotting (11th input argument = 0, default is 1)
% 
% Example 14:  (Version 3.9 or later)
% Exponentially broadened Lorentzian with position=9, height=1.
% x=[0:.1:20]; 
% L=lorentzian(x,9,1);
% L1=ExpBroaden(L',-10)+0.02.*randn(size(x))';
% [FitResults,FitError]=peakfit([x;L1'],0,0,1,18,10)
%
% Example 15: Fitting humps function with two Voigts (version 7.45)
% FitResults,FitError]=peakfit(humps(0:.01:2),60,120,2,20,1.7,5,0)
% FitResults =
%             1       31.099       94.768       19.432       2375.3
%             2       90.128       20.052       32.973       783.34
% FitError =
%       0.72829      0.99915
%
% Example 16: (Version 4.3 or later) Set +/- mode to 1 (bipolar)
% >> x=[0:.1:10];y=exp(-(x-5).^2)-.5*exp(-(x-3).^2)+.1*randn(size(x));
% >> peakfit([x' y'],0,0,2,1,0,1,0,0,0,1,1)
% FitResults =
%             1       3.1636      -0.5433         1.62      -0.9369
%             2       4.9487      0.96859       1.8456       1.9029
% FitError =
%        8.2757
%
% Example 17: Version 5 or later. Fits humps function to a model consisting 
% of one Pearson (shape=4, extra=3) and one Gaussian (shape=1), flat
% baseline mode=3, NumTrials=10.
% x=[0:.005:1.2];y=humps(x);[FitResults,FitError]=peakfit([x' y'],0,0,2,[2 1],[0 0])
% FitResults =
%             1      0.30154       84.671      0.27892       17.085
%             2      0.88522       11.545      0.20825       2.5399
% Baseline =
%         0.901
% FitError =
%        10.457
%
% Example 18: 5 peaks, 5 different shapes, all heights = 1, widths = 3.
% x=0:.1:60;
% y=modelpeaks2(x,[1 2 3 4 5],[1 1 1 1 1],[10 20 30 40 50],...
% [3 3 3 3 3],[0 0 0 2 -20])+.01*randn(size(x));
% peakfit([x' y'],0,0,5,[1 2 3 4 5],[0 0 0 2 -20])
%
% Example 19: Minimum width constraint (13th input argument)
% x=1:30;y=gaussian(x,15,8)+.05*randn(size(x));
% No constraint:
% peakfit([x;y],0,0,5,1,0,10,0,0,0,1,0,0);
% Widths constrained to values above 7:
% peakfit([x;y],0,0,5,1,0,10,0,0,0,1,0,7);
%
% Example 20: Noise test with peak height = RMS noise = 1.
% x=[-5:.02:5];y=exp(-(x).^2)+randn(size(x));P=peakfit([x;y],0,0,1,1,0,10,0,0,0,1,1);
%
% Example 21: Gaussian peak on strong sloped straight-line baseline, 2-peak
% fit with variable-slope straight line (shape 26, peakfit version 6 only).
% >> x=8:.05:12;y=x+exp(-(x-10).^2);
% >> [FitResults,FitError]=peakfit([x;y],0,0,2,[1 26],[1 1],1,0)
% FitResults =           
%             1           10            1       1.6651       1.7642
%             2        4.485      0.22297         0.05       40.045
% FitError =0.093
%
% Example 22: Segmented linear fit (Shape 29, peakfit version 6 only):
% x=[0.9:.005:1.7];y=humps(x);
% peakfit([x' y'],0,0,9,29,0,10,0,0,0,1,1)
%
% Example 23: Polynomial fit (Shape 27, peakfit version 6 only)
% x=[0.3:.005:1.7];y=humps(x);y=y+cumsum(y);
% peakfit([x' y'],0,0,4,1,6,10,0,0,0,1,1)
%
% Example 24: Effect of quantization of independent (x) and dependent (y) variables.
% x=.5+round([2:.02:7.5]);y=exp(-(x-5).^2)+randn(size(x))/10;peakfit([x;y])
% x=[2:.01:8];y=exp(-(x-5).^2)+.1.*randn(size(x));y=.1.*round(10.*y);peakfit([x;y])
%
% Example 25: Variable-alpha Voigt functions, shape 30. (Version 7.45 and
% above only). FitResults has an added 6th column for the measured alphas
% of each peak.
% x=[0:.005:1.3];y=humps(x);[FitResults,FitError]=peakfit([x' y'],60,120,2,30,15)
% FitResults =
%             1      0.30147       95.797      0.19391       24.486       2.2834
%             2      0.89186       19.474      0.33404       7.3552      0.39824
% FitError =
%       0.84006      0.99887
%
% Example 26: Variable time constant exponentially braodened Gaussian
% functions, shape 31. (Version 7 and above only). FitResults has an added
% 6th column for the measured  time constant of each peak.
% [FitResults,FitError]=peakfit(DataMatrix3,1860.5,364,2,31,32.9731,5,[1810 60 30 1910 60 30])
% FitResults =
%        1       1800.6       1.8581       60.169       119.01       32.781
%        2       32.781      0.48491       1900.4        30.79       33.443
% FitError =
%      0.076651      0.99999
%
% Example 27 Pearson variable shape, PeakShape=32,(Version 7 and above
% only). Requires modelpeaks2 function in path.
% x=1:.1:30;y=modelpeaks2(x,[4 4],[1 1],[10 20],[5 5],[1 10]);
% [FitResults,FitError]=peakfit([x;y],0,0,2,32,10,5)
%
% Example 28 Gaussian/Lorentzian blend variable shape, PeakShape=33
% (Version 7 and above only). Requires modelpeaks2 function in path.
% x=1:.1:30;y=modelpeaks2(x,[13 13],[1 1],[10 20],[3 3],[20 80]);
% [FitResults,FitError]=peakfit([x;y],0,0,2,33,0,5)
%
% Example 29:  Fixed-position Gaussian (shape 16), positions=[3 5]. 
% x=0:.1:10;y=exp(-(x-5).^2)+.5*exp(-(x-3).^2)+.1*randn(size(x));
% [FitResults,FitError]=peakfit([x' y'],0,0,2,16,0,0,0,0,[3 5])
%

