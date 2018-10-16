% Matlab script for random matrix exercise

% Generate M members of the GOE ensemble of NxN matrices, and 
% calculate the difference between the eigenvalues 
% in the middle of the range

% Increase M, N, and Nbins as appropriate

M = 1000;
N = 100;
Nbins = 100;

% randn gives Gaussian random numbers; randn(N) generates a NxN matrix
% Mat' is the transpose

differences = zeros(M,1);	% Initialize array of differences
M11 = zeros(M,1);	% Initialize array of 11 entries

for m = 1:M
    Mat = randn(N);
    Ms = Mat + Mat';
    lambda = sort(eig(Ms));
    differences(m) = lambda(N/2+1)-lambda(N/2);
    M11(m) = Ms(1,1);
end

% Divide out by mean value of the splittings
diffAve = mean(differences);

% hist generates a histogram of differences with Nbins bins. 
% Change Nbins to 50 or so when you do production runs. 

% Octave allows one to use the third argument to normalize the area of
% the histogram (say, to one).
% Matlab doesn't normalizes the histograms 

hist(differences/diffAve,Nbins)

% PLOTTING THEORY OVER HISTOGRAM

% Demonstrated with histogram for diagonal element
% First plot histogram for M11
subplot(1,2,1)
hist(M11,Nbins)

% "hold on" and "hold off" allow one to combine curves
hold on

% Now, plot expected Gaussian fit

% define x curve
x = min(M11)-1:0.01:max(M11)+1;
M11Range = max(M11)-min(M11)

% y(x) is Gaussian of RMS width sigma = 2 (diagonal element doubled in size)
% expected width of diagonal element
sigma11 = 2;
y = (1/sqrt(2*pi*sigma11^2))*exp(-x.*x/(2*sigma11^2));

% Histogram multiplies height by number of entries, divides it by Nbins
normalization = M*M11Range/Nbins
plot(x,normalization*y,'k')

% Reset graphics so next curve is not overlayed on these two
hold off

% GENERATING RANDOM +-1 MATRICES

% Symmetric matrix with integer values +-1 with 50/50 probability
% rand gives flat distributions of random numbers between 0 and 1
% generate random matrix of +- 1
MatPM = sign(2*rand(N) -1)

% Allocate space for NxN matrix Ms
MsPM = zeros(N);

% Symmetrize: copy top half of the matrix (diagonal copied twice)
for i=1:N
  for j=i:N
    MsPM(i,j) = MatPM(i,j);
    MsPM(j,i) = MatPM(i,j);
  end
end  

% Show the matrix: check that it worked
MsPM

%part D
s = 0:.0001:5;
pwig = (pi*s/2).*exp(-pi*(s.^2)/4);

subplot(1,2,2)
plotyy(s,pwig,x,normalization*y)
%hold on
%plot(x,normalization*y,'k')


xlabel('s')
ylabel('\rho_{Wigner} (s)')