%% Configure network 
% Set input matrix (x) and target matrix (t) for ANN fitting
% if matrix rows are samples, use .' (i.e. x = inputs.')
% if columns are samples, omit .'

x = input_mat1.'; %input matrix
t = target_mat1.'; %target matrix

trainFcn = 'trainlm'; % set training type (type: help nntrain to see all)
net = fitnet([6,3,2],trainFcn); % set number of neurons in each hidden layer as a row vector


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
view(net) % display diagram of network

%% Test network using new data
% Test the Network
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