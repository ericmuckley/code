%% Classify app

skip = 10;       %number of points to skip
sampnum = 8;    %number of samples 

%define classes
b = cell(length(bt),1); b(:) = {'boron'};
cb = cell(length(bt),1); cb(:) = {'cb'};
g = cell(length(bt),1); g(:) = {'g'};
h5 = cell(length(bt),1); h5(:) = {'h5'};
one = cell(length(bt),1); one(:) = {'one'};
five = cell(length(bt),1); five(:) = {'five'};
seven = cell(length(bt),1); seven(:) = {'seven'};
t1150 = cell(length(bt),1); t1150(:) = {'t1150'};

classnames = vertcat(b,cb,g,h5,one,five,seven,t1150);

%one long row vector of the 0-7 classes
classes0 = [ones(1,length(bt)).*0 ones(1,length(bt)) ones(1,length(bt)).*2 ...
    ones(1,length(bt)).*3 ones(1,length(bt)).*4 ones(1,length(bt)).*5 ...
    ones(1,length(bt)).*6 ones(1,length(bt)).*7] ;
%make the row vector into a column vector for the classify app
classes = transpose(classes0);
%define the three predictore variables 
o2 = vertcat(bo2, cbo2, go2, h5o2, oneo2, fiveo2, seveno2, t1150o2);
co2 = vertcat(bco2, cbco2, gco2, h5co2, oneco2, fiveco2, sevenco2, t1150co2);
temp = vertcat(bt, cbt, gt1, h5t, onet, fivet, sevent, t1150t);


%shorten the predictor variables by factor of 'skip' 
o2short = o2(1:skip:length(o2));
co2short = co2(1:skip:length(co2));
tempshort = temp(1:skip:length(temp));
classshort = classes(1:skip:length(classes));
classnamesshort = classnames(1:skip:length(classnames));
predvarsshort = [o2short co2short tempshort];
predvarsshortstr = num2str([o2short co2short tempshort]);
%put predictors and classes into single matrix for classify app
all = [o2short co2short tempshort classshort];

%allwithnames = cell(length(classnamesshort),4);


% best score: multiple classifiers >99%
% with all 3 inputs, no PCA, with skip = 7, 5-fold cross-validation



%{
allwithnames = {};
for aa = 1:length(classnamesshort)
    allwithnames{aa} = [classnamesshort(aa); num2str(predvarsshort(:,aa))]'; 
end
%}


%{ 
X = nninputs;
Y = classshort;
ctree = fitctree(X,Y);
view(ctree) 
view(ctree,'mode','graph')

;

%}

%% NN APP
%make matrix of just predictor variables for NN app
nninputs = [o2short co2short tempshort];
%make class matrix for NN app
col0 = transpose(zeros(1,length(nninputs)/sampnum));
col1 = col0 + 1;
classmat  = [vertcat(col1,col0,col0,col0,col0,col0,col0,col0)...
    vertcat(col0,col1,col0,col0,col0,col0,col0,col0)...
    vertcat(col0,col0,col1,col0,col0,col0,col0,col0)...
    vertcat(col0,col0,col0,col1,col0,col0,col0,col0)...
    vertcat(col0,col0,col0,col0,col1,col0,col0,col0)...
    vertcat(col0,col0,col0,col0,col0,col1,col0,col0)...
    vertcat(col0,col0,col0,col0,col0,col0,col1,col0)...
    vertcat(col0,col0,col0,col0,col0,col0,col0,col1)];
    
