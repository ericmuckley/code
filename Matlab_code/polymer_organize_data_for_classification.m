% delete classes so transpose doesn't happen on old variables
clear class0 class1 class13 class14 all_classes



classnum = 4; % set number of classes 
samplenum = sum(length(wavelength_ex)+length(wavelength_abs)+length(wavelength_em));
% samplenum = 3475; %tot number of samples (sum of all wavelengths used?)
fill0 = zeros(samplenum, 1); %create filler zeros to fill up class matrix
fill1 = ones(samplenum, 1); %create filler ones to fill up class matrix

% make 4 column vectors for classes 0,1,13,14
class0(1:samplenum) = 0;
class0 = transpose(class0);
class1(1:samplenum) = 1;
class1 = transpose(class1);
class13(1:samplenum) = 13;
class13 = transpose(class13);
class14(1:samplenum) = 14;
class14 = transpose(class14);


all_classes = vertcat(class0, class1, class13, class14); % make single target column of all target classes
class_mat = [fill1, fill0, fill0, fill0; fill0, fill1, fill0, fill0; ... %make single target matrix for all classes
    fill0, fill0, fill1, fill0; fill0, fill0, fill0, fill1];   

%create input matrix
inputs = [wavelength_abs, H0abs; wavelength_ex, H0ex; wavelength_em, H0em; wavelength_abs, H1abs; wavelength_ex, H1ex; wavelength_em, H1em]
%{
wavelength_ex, H0ex; wavelength_em, H0em; ...
    wavelength_abs, H1abs; wavelength_ex, H1ex; wavelength_em, H1em; ...
    wavelength_abs, H13abs; wavelength_ex, H13ex; wavelength_em, H13em ...
    wavelength_abs, H14abs; wavelength_ex, H14ex; wavelength_em, H14em]
%}



