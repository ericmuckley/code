addpath('Tools');
addpath('Simulated_Data');
addpath('Regularization');

%startup;
 
% -------------------------------------------------------------------------
% Copyright 2008. Thorsten Hohage, Klaus Giewekemeyer, Tim Salditt.
% 
% REFREG is free software. If you wish to publish work using this or part
% of this software, please meake a reference on the source of the code.
% 
% Under usage of the freely available routines RANDRAW.M (by Alex Bar Guy &
% Alexander Podgaetsky) and CUMSIMPSON.M (by D.C. Hanselman) [1].
% 
% References:
% 
% [1] The files RANDRAW.M and CUMSIMPSON.M have been
%     downloaded from the Matlab Central file exchange:
%     http://www.mathworks.com/matlabcentral/fileexchange/
% -------------------------------------------------------------------------
% global constants

% classical electron radius
r_0 = 2.817940289*1E-5; % Angstrom

%--------------------------------------------------------------------------
% Data input

%input_file = 'ifortythtree.txt';
%[q_exp,y,y_err] = textread(input_file)%,'%f %f %f', 'delimiter','\t','commentstyle','shell');

q_expp = A43;
q_exp = 2*pi/1.54.*sin(q_expp);
y = B43;
y_err = C43;


% first point is excluded
q_exp = q_exp(2:end);
y = y(2:end);
y_err = y_err(2:end);

%--------------------------------------------------------------------------
% parameter declaration

% length of support [A]
a = 1100;
% lower bound of support [A]
left_support = 0;
% length of the apriori known part of the profile [A]
len_apriori =  23;
% lower bound on the length of the interval left of the known jump [A]
left_apriori = -10;
% number of points in q-space (without later-added theoretical point (q=0,y=1))
m = length(q_exp);
%r = q_max*a/n (accuracy parameter for trapezoidal rule), n is defined by
%this equation
r = 0.25;
% number of unkonwns in z-space
n = ceil(q_exp(end)*a/r);
% corresponding number of sampling points for trapezoidal rule
N = 2^nextpow2(n)+1;

% sampling distance of points in real space (just dependent on r)
dz = a/(n-1);
% z-vector [A]
z_vek = dz*[0:1:n-1]' + left_support;

% distance of points in q-space [1/A]
dq = 2*pi*n/(a*N);
% equidistant q-vector for zero-padding (length N) [1/A]
q_vek = dq*[-N/2:1:N/2]';
% mirrored abscissa-data-vector (length 2m+1) [1/A]
q_exp = [-q_exp(end:-1:1);0;q_exp];
% mirrored data vector [counts]
y = [y(end:-1:1);1;y];
% mirrored data error vector [counts]
% Note: A relative error of 5% is added to the point at the origin,
% reflecting an upper bound of the uncertainty for the estimation of 
% the primary beam intensity, which is used to normalized the data to
% the plateau of total external reflection.
y_err = [y_err(end:-1:1);0.05;y_err];

% degree of polynomial for interpolation matrix
polynom_grad = 5;
% interpolated equi-spaced data in q-space
J = interpolation_matrix(N,dq,q_exp,polynom_grad);

% parameter for abort of Newton iteration by discrepancy principle;
% minimum reduced chi^2 value to be expected (must be >2)
tau = 2.7;
% maximum number of Newton iterations
N_GN = 20;
% rho for abort criterion of CG iteration
rho = 0.8;
% maximum number of CG iterations
N_CG = 50;

% summary
param.a = a;
param.left_support = left_support;
param.len_apriori = len_apriori;
param.left_apriori = left_apriori;
param.dq =dq;
param.m =m;
param.N=N;
param.dz = dz;
param.n = n;       
param.J = J;
param.tau = tau;
param.N_GN = N_GN;
param.rho = rho;
param.N_CG = N_CG;
param.y_err = y_err;
param.z_vek = z_vek; 

%--------------------------------------------------------------------------
% parameters of initial guess in the box model

% For M layers: [sigma_0 (Angstrom), rho_1, d_1, sigma_1, ..., sigma_M]
% with [sigma_i] = Angstrom, [rho_i] = number of electrons/Angstroem^3, [d_i] = Angstrom
% Nomenclature: 
%
%   sigma_0:             roughness of substrate/first-layer interface
%   [rho_i,d_i,sigma_i]: electron density, thickness, roughness of i-th
%                        interface

% initial guess
Param_guess = [50.0000    0.6380   500    50    ];%0.3290    3.5700    3.2770    0.4510    5.2420    2.4610];
               
% substrate density [number of electrons/A^3](fixed by user)
rho_sub = 0.69;
% bulk density [number of electrons/A^3] (fixed by user)
rho_bulk = 0.334;

% display parameters
%param;

%--------------------------------------------------------------------------
% Calculation of the start profile

% initial guess for normalized electron density
rho_start = rho_ana(z_vek,Param_guess,rho_sub,rho_bulk);
% corresponding derivative of the electron density
phi_start = derivative(z_vek,rho_start);
% corresponding calculated Fresnel-normalized reflectivity
y_start = Fforward(phi_start,param);

%--------------------------------------------------------------------------
% Reconstruction of the profile

% derivative of reconstructed normalized electron density
dummy = Newton_CG(phi_start, y, param);
phi_recon = dummy{1,1}; red_chi_sq = dummy{1,2};xi = dummy{1,3};
dummy = cumsimpson(z_vek,phi_recon)+1;
% reconstructed absolute electron density
rho_recon = (rho_sub-rho_bulk)*dummy + rho_bulk;

% modelled normalized reflectivity corresponding to reconstruction
y_recon = Fforward(phi_recon,param);


%--------------------------------------------------------------------------
% reconstructed electron density

subplot(3,2,[1,3]);
hold on;
%plot area of apriori information
z_v_apr = z_vek(find(z_vek >= left_apriori & z_vek <= left_apriori+len_apriori));
p0 = area(z_v_apr,z_v_apr*0+0.75,'FaceColor',[0.9,0.9,0.9]);
set(gca,'Layer','top');
%plot profiles
rho_start = (rho_sub-rho_bulk)*rho_start + rho_bulk;
Delta_z = 0;
P = plot(z_vek+Delta_z,rho_recon,z_vek,rho_start);
legend([P;p0],'Reconstruction','Initial guess','Known interval');
xlabel(['z [',char(197),']']);
ylabel(['\rho_e(z) [e^{-}/',char(197),'^{3}]']);

xlim([left_support left_support+a]);
ylim([0.2 0.75]);
hold off;
% doublearrow for apriori-interval
% This function uses the m-file dsxy2figxy, which is provided in the help
% of the Matlab software package. This file is not contained in the REFREG
% files.
% x1 = left_apriori;
% y1 = 0.25;
% x2 = left_apriori + len_apriori;
% y2 = 0.25;
% [arrowx,arrowy] = dsxy2figxy(gca,[x1;x2],[y1;y2]);
% annotation('doublearrow',arrowx,arrowy);

%--------------------------------------------------------------------------
% Comparison exp./modelled reflectivity

subplot(3,2,[2,4]);
hold on;

p2 = errorbar(q_exp,y,y_err,'ko');
p1 = plot(q_exp,y_recon,'r');
legend([p1;p2],'Reconstruction','Experimental data');
xlim([0 q_exp(end)]);
xlabel(['q [',char(197),'^{-1}]']);
ylabel('R(q)/R_F(q)');
hold off;

%--------------------------------------------------------------------------
% Residuals

subplot(3,2,6);
res = (y-y_recon)./y_err;
hold on;
plot(q_exp,res,'o-');
plot(q_exp,0*q_exp+1,'r--');
plot(q_exp,0*q_exp-1,'r--');
xlim([0 q_exp(end)]);
ylim([-1000 1000]);
xlabel(['q [',char(197),'^{-1}]']);
ylabel('[y_j^\delta-r(q_j)]/\delta_j');
hold off;

%--------------------------------------------------------------------------
% red_chi_sq

subplot(3,2,5);
plot(red_chi_sq,'o-');
set(gca,'YScale','log');
xlabel('Newton-CG-Iteration');
ylabel('||F(\phi^{(k)}) - y^\delta||_Y^2');

rmpath('Tools');
rmpath('Simulated_Data');
rmpath('Regularization');

hold off