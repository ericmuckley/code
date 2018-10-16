%% water thickness part
kBT = 0.0256; %eV
v = 29.88; %Angstrom^3
A = -0.4; %eV
p0 = max(press);


d = (A*v ./ (6*pi*kBT.*(log(press./p0)))).^(1/3);

a = .195;
d_adjust = a*d; 
%% fitting part

R0 = min(Rm);
D = .00001;
B1 = 15;
B2 = .7;
B3 = .95;

v1 = B2*(1+tanh(.3.*(press-10.3))).*d;
v2 = B3*(-1-tanh(.2.*(press-19.1))).*d;

fit = R0 + D.*sqrt((freq)).*B1.*(d).^(3/2) + v1 + v2;


%% plotting part
     
plot(press, Rm, 'k')
hold on   
plot(press, fit)
%plot(press,v1)
%plot(press,v2)

xlabel('H_2O Pressure (Torr)')
ylabel('R_m (\Omega)')
legend('Measured R_m', 'Fit')
set(gca,'FontSize',18);
axis([0 30 0 30])
hold off