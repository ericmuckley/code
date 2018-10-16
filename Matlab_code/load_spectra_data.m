% Put all .txt files in Matlab directory then run script to import them as
% variables

% Import absorption (abs) data
absH0 = importdata('absH0.txt', '\t' ,1); %load H0 absorption data
absH1 = importdata('absH1.txt', '\t' ,1); %load H1 absorption data
absH13 = importdata('absH13.txt', '\t' ,1); %load H13 absorption data
absH14 = importdata('absH14.txt', '\t' ,1); %load H14 absorption data
% Redefine abs data to remove headers
absH0 = (absH0.data);
absH1 = (absH1.data);
absH13 = (absH13.data);
absH14 = (absH14.data);
% Redefine abs data to omit wavelengths outside 300-800 nm range
absH0 = absH0(2201:2701,:);
absH1 = absH1(2201:2701,:);
absH13 = absH13(2201:2701,:);
absH14 = absH14(2201:2701,:);

% Import photoluminescence (pl) data
plH0 = importdata('plH0.txt', '\t' ,1); %load H0 pl data
plH1 = importdata('plH1.txt', '\t' ,1); %load H1 pl data
plH13 = importdata('plH13.txt', '\t' ,1); %load H13 pl data
plH14 = importdata('plH14.txt', '\t' ,1); %load H14 pl data
% Redefine pl data to remove headers
plH0 = (plH0.data);
plH1 = (plH1.data);
plH13 = (plH13.data);
plH14 = (plH14.data);

% Import excitation (ex) data
exH0 = importdata('exH0.txt', '\t' ,1); %load H0 ex data
exH1 = importdata('exH1.txt', '\t' ,1); %load H1 ex data
exH13 = importdata('exH13.txt', '\t' ,1); %load H13 ex data
exH14 = importdata('exH14.txt', '\t' ,1); %load H14 ex data
% Redefine ex data to remove headers
exH0 = (exH0.data);
exH1 = (exH1.data);
exH13 = (exH13.data);
exH14 = (exH14.data);

% To select data for a specific temperature, select the correct column
% corresponding to that temperature. In each matrix, the columns are
% defined as:
% column 1 = wavelength (nm)     column 2 = intensity @ 10C
% column 3 = intensity @ 15C     column 4 = intensity @ 20C
% column 5 = intensity @ 25C     column 6 = intensity @ 30C
% column 7 = intensity @ 35C     column 8 = intensity @ 40C
% column 9 = intensity @ 45C     column 10 = intensity @ 50C
% column 11 = intensity @ 55C    column 12 = intensity @ 60C
% column 13 = intensity @ 65C    column 14 = intensity @ 70C
% column 15 = intensity @ 75C    column 16 = intensity @ 80C

wl_abs = absH0(:,1); % extract absorption wavelengths
wl_pl = plH0(:,1); % extract pl wavelengths
wl_ex = exH0(:,1); % extract ex wavelengths

% Create temperature column vectors, where t_n = length of desired vector
t_n = 671; 
t10(1:t_n,1) = 10;    t15(1:t_n,1) = 15;    t20(1:t_n,1) = 20;
t25(1:t_n,1) = 25;    t30(1:t_n,1) = 30;    t35(1:t_n,1) = 35;
t40(1:t_n,1) = 40;    t45(1:t_n,1) = 45;    t50(1:t_n,1) = 50;
t55(1:t_n,1) = 55;    t60(1:t_n,1) = 60;    t65(1:t_n,1) = 65;
t70(1:t_n,1) = 70;    t75(1:t_n,1) = 75;    t80(1:t_n,1) = 80;








