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
B2 = 2.55;
B3 = 6;

v1 = B2*(1+tanh(.3.*(press-10.9)));
v2 = B3*(-1-tanh(.14.*(press-23)));

fit = R0 + D.*sqrt((freq)).*B1.*(d).^(3/2) + v1 + v2;
thickness_contribution = R0 + D.*sqrt((freq)).*B1.*(d).^(3/2) ;


%% plotting part
subplot(1,2,1)     
plot(press, Rm, 'k')
hold on 
plot(press,thickness_contribution)
plot(press, fit)
xlabel('H_2O Pressure (Torr)')
ylabel('R_m (\Omega)')
legend('Measured R_m', 'Contribution from H_2O adsorption on film surface', 'Sum of all contributions')
set(gca,'FontSize',15);
axis([0 30 0 30])
hold off

subplot(1,2,2)     
plot(press, v1)
hold on
plot(press,v2)
legend('Contribution from H_2O diffusion into bulk film','Contribution from film delamination from substrate')
xlabel('H_2O Pressure (Torr)')
ylabel(' \Delta R_m (\Omega)')
set(gca,'FontSize',15);

%plot(press,v1)
%plot(press,v2)


hold off