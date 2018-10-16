%% water thickness part

kBT = .0256;
v = 30; %Angstrom^3
H = -.4;
p0 = 25.8;

p = .01:.01:p0; %pressure range 

%d = v*((H)./(6*pi*kBT.*log(p/p0))).^(1/3); %angstrom

a = .2;

%model: R_m = a * d + c
%where a = density, freq; c = internal and mounting crystal frictions

%% plot piecewise

%segment boundaries:
sb1 = 8.75;
sb2 = 13.5;
sb3 = 16;
sb4 = 22.5;
%viscosity change factors:
v1 = .5;
v2  = -.5;

fit=nan*ones(size(p)); %define fit
% compute each segment
%1st segment (before hump)
fit(p<sb1)= a*v*((H)./(6*pi*kBT.*log(p(p<sb1)/p0))).^(1/3);    
%2nd segment (rising hump)
fit(sb1<=p&p<=sb2)= a*v*((H)./(6*pi*kBT.*log(p(sb1<=p&p<=sb2)/p0))).^(1/3) + v1*p(sb1<=p&p<=sb2) - v1*sb1;
%3rd segment (plateau of hump)
fit(sb2<=p&p<=sb3)= a*v*((H)./(6*pi*kBT.*log(p(sb2<=p&p<=sb3)/p0))).^(1/3) + v1*p(sb2<=p&p<=sb3) - v1*sb2;
%4th segment (decreasing hump)
fit(sb3<=p&p<=sb4)= a*v*((H)./(6*pi*kBT.*log(p(sb3<=p&p<=sb4)/p0))).^(1/3) + v2*p(sb3<=p&p<=sb4) - v2*sb3;
%5th segment 

%4rd segment     
fit(sb3<p)= p(sb3<p);  %3rd part

     %% plot
     
     
plot(H2O_pressure_torr, resistance_ohm, 'k')
hold on   
plot(p,dnorm)
plot(p, fit)

xlabel('H_2O Pressure (Torr)')
ylabel('R_m (Angstrom)')

legend('R_m', 'thickness', 'fit')
set(gca,'FontSize',15);

   
hold off
     