%% Prepare image
clearvars
tic;

rawI = (imread('ps20.jpg')); %read in image
grayI = double(rgb2gray(rawI)); %convert raw image to grayscale
I_size = size(grayI); %get size of image
numrows_in_I = I_size(1); %get unmber of rows in image
numcolumns_in_I = I_size(2); %get number of columns in image

pixelnumber = zeros(numrows_in_I,1);
for i = 1:numcolumns_in_I; %get pixel numbers
    pixelnumber(i) = i;
end

subplot(2,2,1) %show raw image in first subplot
imshow(rawI)

%% Set fit data/parameters: 'Gausss CNT fit'.

ft = fittype( 'gauss8' );                               %fit to 8 Gaussian peaks
opts = fitoptions( 'Method', 'NonlinearLeastSquares' ); %fit type is nonlinear least squares method
%opts.Display = 'Off';
%these are orgnaized as [a1 b1 c1 a2 b2 c2 a3...] where fi(x) = ai*exp(-((x-bi)/ci)^2)
opts.Lower = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; %lower limits for ai,bi,ci
opts.StartPoint = [251 66 12.9399038849393 171.343891755542 89 20.3848242321851 169.327617517717 34 28.7943277331627 167.999924867436 167 20.0322334038051 167 458 18.3478183128687 161.999999999999 282 10.2986481060411 143.466360077877 303 12.7841873395802 137.999809896395 243 17.4516975802209]; %starting points for ai,bi,ci
opts.Upper = [250 213 206 250 513 206 250 513 206 250 513 206 250 513 206 250 513 206 250 513 206 250 513 206]; %upper limits for ai,bi,ci, 
%I use length and mean for upper limits

%% Loop over image columns
num_of_columns_to_scan = 3;
avgcolumnheight = zeros(num_of_columns_to_scan,1);
avgcolumnwidth = zeros(num_of_columns_to_scan,1);
for i = 1:num_of_columns_to_scan
    Xvar = pixelnumber;     %input x data  
    Yvar = grayI(:,230+i);        %input y data  
    [xData, yData] = prepareCurveData(Xvar, Yvar); %format fitting data
  
    [fitresult, gof] = fit( xData, yData, ft, opts ); % Fit model to data

    subplot(2,2,i+1)
    plot( fitresult, xData, yData);
    legend off
    title(' ')
    xlabel('Pixel number') % Label axes
    ylabel(['Pixel value column ' num2str(i)])
    axis([0 max(Xvar) 0 1.1*max(Yvar)])
    
    fitparameters = coeffvalues(fitresult);
    avgcolumnheight(i) = mean([fitparameters(1) fitparameters(4) fitparameters(7) fitparameters(10) fitparameters(13) fitparameters(16) fitparameters(19) fitparameters(22)]);
    avgcolumnwidth(i) = mean([fitparameters(3) fitparameters(6) fitparameters(9) fitparameters(12) fitparameters(15) fitparameters(18) fitparameters(21) fitparameters(24)]);
    
end

avgIheight = mean(avgcolumnheight);
avgIwidth = mean(avgcolumnwidth);

execution_time = toc;