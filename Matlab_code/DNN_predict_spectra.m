%% Organize data

% omit wavelengths other than those from 300-800 nm 
%input_mat1NEW = input_mat1([2201:2701],:)

% import H1abs_alltempts.txt with each column as a variable and change
% wavelengthnm to 'w'
snum = length(w); % find number of samples per temperature 

i10(1:snum) = 10; % identify temp as variable
i15(1:snum) = 15;
i20(1:snum) = 20;
i25(1:snum) = 25;
i30(1:snum) = 30;
i35(1:snum) = 35;
i40(1:snum) = 40;
i45(1:snum) = 45;
i50(1:snum) = 50;
i55(1:snum) = 55;
i60(1:snum) = 60;
i65(1:snum) = 65;
i70(1:snum) = 70;
i75(1:snum) = 75;
i80(1:snum) = 80;

input_mat1 = [w, i10.'; w, i15.'; ... % this is marix for training (T = 10-80 w/0 50C)
             w, i20.'; w, i25.'; ...
             w, i30.'; w, i35.'; ...
             w, i40.'; w, i45.'; ...
           %  w, i50.'; ...
             w, i55.'; ...
             w, i60.'; w, i65.'
             w, i70.'; w, i75.'
             w, i80.'];

target_mat1 = [t10; t15; t20; t25; t30; t35; t40; t45; ...
    %t50;...
    t55; t60; t65; t70; t75; t80]; %this is matrix for targets (T = 10-65C)





%% Configure deep network 
% Set input matrix (x) and target matrix (t) for ANN fitting
% if matrix rows are samples, use .' (i.e. x = inputs.')
% if columns are samples, omit .'

x = input_mat1.'; %input matrix
t = target_mat1.'; %target matrix

trainFcn = 'trainlm'; % set training type (type: help nntrain to see all)
net = fitnet([12,9,6,3],trainFcn); % set number of neurons in each hidden layer as a row vector


% Input/Output Pre/Post-Processing (type: help nnprocess) 
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing (type: help nndivide)
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = .75; % percent to use for training
net.divideParam.valRatio = .15; % percent to use for validation
net.divideParam.testRatio = .1; % percent to use for testing
net.performFcn = 'mse'; % Choose a Performance Function (type help nnperformance to see all)

% Choose Plot Functions (type: help nnplot)
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotregression', 'plotfit'};

[net,tr] = train(net,x,t);
%view(net) % display diagram of network

%% Test network
y = net(x); % run the network, where y are predictions and x are inputs
e = gsubtract(t,y); % find raw errors between targets and predictions
performance = perform(net,t,y) %display MSE value of training errors



%% Recalculate Training, Validation and Test Performance
%{
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)
%}
%% Plot results (uncomment lines to enable plots)
%{
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)
%}
%% Deployment
%{
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false) % view values of biases and weights
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end
%}


%% Test on new data and plot results

input50 = [w, i50.']; % make input matrices for predicting (T = 50C)

predict50 = net(input50.');

plot(w,predict50)
hold on
plot(w,t50)
legend('prediction', 'actual')



%% Close 
clear i10 i15 i20 i25 i30 i35 i40 i45 i50 i55 i60 i65 i70 i75 i80
