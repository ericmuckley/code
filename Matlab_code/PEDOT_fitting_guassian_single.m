%% water thickness part

kBT = .0256;
v = 30; %Angstrom^3
H = -.4;
p0 = 25.8;

p = .01:.01:p0; %pressure range 

d = v*((H)./(6*pi*kBT.*log(p/p0))).^(1/3); %angstrom

a = .195;

%model: R_m = a * d + c
%where a = density, freq; c = internal and mounting crystal frictions
visc_center1 = 15;
visc1  = -4+ 5*exp(-.02*(p-visc_center1).^2);

%visc_center2 = 28;
%visc2  = -10*exp(-.03*(p-visc_center2).^2);

fit = a*d + visc1 + 4- .2*p ;


     
plot(H2O_pressure_torr, resistance_ohm, 'k')
hold on   
plot(p,dnorm)
plot(p, fit)

xlabel('H_2O Pressure (Torr)')
ylabel('R_m (Angstrom)')

legend('R_m', 'thickness', 'fit')
set(gca,'FontSize',15);

   
hold off
     