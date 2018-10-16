%plot the actual variables alongside the predictions
%{
plot(chamberpressureTorrc)
hold on
plot(output_1cc5)
%plot(output_2*0.238)
legend('Actual pressure','Predicted Ar Pressure','Predicted H_2O Pressure')
xlabel('Time')
ylabel('Pressure (Torr)')
hold off
%}

plot(chamberpressureTorrc)
hold on
%plot((chamberpressureTorrc - output_1cc5))
axis([0 6000 -8 110])
plot(error_Pcc5)
plot
hold off