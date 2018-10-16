
kBT = .0256;
v = 30; %Angstrom^3
H = -.4;
p0 = 23.8;


p = 0:.01:23.8;

d = v*((H)./(6*pi*kBT.*log(p/p0))).^(1/3);
plot(p,d)
xlabel('H_2O Pressure (Torr)')
ylabel('H_2O thickness (Angstrom)')