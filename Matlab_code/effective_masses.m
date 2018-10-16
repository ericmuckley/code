clear all

Eh = -.5991:.0001:-.599; %eV

V0 = 1; %eV
a = 3; %angstrom
w = 1.5; %angstrom
kBZ = 1.0472; %angstrom^(-1)
hbar = 6.582119E-16; %eV s
me = 9.10938E-31; %kg

%%
kh = sqrt((2*me*Eh)./(hbar^2));
kph = sqrt((2*me*(Eh+V0))./(hbar^2));



C1 = cos(kh.*(a-w)).*cos(kph.*w);
C2 = -(1/2)*((-1j*kph./kh)-(1j*kh./kph)).*sin(kh.*(a-w)).*sin(kph.*w)*1j;
Kh= acos(C1+C2)./a;
   
%%
Kminussquare = (K - kBZ).^2;
plot(Kminussquare,Eh,'k')
xlabel( '$(K - k_{BZ})^2~~ $(\AA$^{-2}$)', 'Interpreter','LaTex');
ylabel('E (eV)');
title('Slope at top of first band')
set(gca,'FontSize',15)
fit = polyfit(Kminussquare,Eh,1)
slope = fit(1)
E_0bottom = fit(2)

effective_mass = hbar^2/(2*slope)

mass_ratio = effective_mass / me


