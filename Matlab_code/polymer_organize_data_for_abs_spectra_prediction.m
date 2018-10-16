% create absorption training matrix for ANN curve fitting
% use H0, H1, H13 to predict abdorption spectrum of H14
% import abs_em_ex_25C.txt file and use each column as a variable to run
% this script
clear A0 A1 A13
% set number of wavelength samples
samplenum = length(wavelength_abs);
% label deuteration numbers
A0(1:samplenum) = 0; 
A1(1:samplenum) = 1;
A13(1:samplenum) = 13;
A14(1:samplenum) = 14;
% line up each spectra with its corresponging deuteration number
inputs0 = [transpose(A0), wavelength_abs]; 
inputs1 = [transpose(A1), wavelength_abs];
inputs13 = [transpose(A13), wavelength_abs];
inputs14 = [transpose(A14), wavelength_abs];
inputs = vertcat(inputs0, inputs1, inputs13); %input matrix
targets = vertcat(H0abs, H1abs, H13abs); % target matrix
% now test network on H14
TESTinputs = inputs14;
TESTtargets = H14abs;

clear A0 A1 A13 inputs0 inputs1 inputs13

