% Fitting with Neural Network

input_data = [QCMfreqHzp QCMresistanceOhmp]; %create input matrix from your input columns
target_data = [chamberpressureTorrp chamberRHp]; %create output matrix from your output columns


x = input_data'; %name your input data set (with ' afterwards) x
t = target_data'; %name your target data set (with ' afterwards) t


% Choose a Training Function (or type help nntrain)
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.


% Create a Fitting Network
hiddenLayerSize = 25; %number of neurons in hidden layer
net = fitnet(hiddenLayerSize,trainFcn);


net.input.processFcns = {'removeconstantrows','mapminmax'}; %input processing function
net.output.processFcns = {'removeconstantrows','mapminmax'}; %output processing function


% Split Data for Training, Validation, Testing (or type help nndivide)
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net5neurons_c(x);
e = gsubtract(t,y);
performance = perform(net,t,y);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);

% View the Network
%view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
option1=false;
option2=false;
option3=false;

if (option1)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (option2) 
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (option3)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end





%ORGANIZE AND PLOT RESULTS:



ytran=y.'; %transpose y matrix
output_1 = ytran(:, 1); %get 1st column of y matrix
output_2 = ytran(:, 2); %get 2nd column of y matrix



iterationnumber=1:1:length(input_data); %make generic x-values for plot as long as the input vectors

%plot the actual variables alongside the predictions
plot_actual = false;
if plot_actual
plot(iterationnumber,chamberpressureTorr)
hold on
plot(iterationnumber,output_1)
plot(iterationnumber,output_2*0.238)
legend('Actual pressure','Predicted Ar Pressure','Predicted H_2O Pressure')
xlabel('Time')
ylabel('Pressure (Torr)')
hold off
end 

hold off


both_pressurep = [chamberpressureTorrp output_1];
mean_pressurep = mean(both_pressurep, 2);
    
both_RHp = [chamberRHp output_2];
mean_RHp = mean(both_RHp, 2);

delta_signalPp = 100*abs(mean_pressurep - output_1)./ round(chamberpressureTorrp,0);
delta_signalRHp = 100*abs(mean_RHp - output_2)./ round(chamberRHp,0);


%plot the errors
plot_error = true;
if plot_error

scatter(chamberpressureTorrp, delta_signalPp, 10,[0 0 0])
hold on
scatter(chamberpressureTorrp, delta_signalRHp,10)
xlabel('Pressure (Torr)')
ylabel('Error (|%|)')
axis([5 110 0 10])
legend('Presidtion of Ar pressure','Prediction of H_20 pressure')
box on
end

