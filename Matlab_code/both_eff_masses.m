clc
clear all

Eh = 3.32:.00001:3.36; %eV
E = 4:.00001:4.05; %eV

V0 = 1; %eV
a = 3; %angstrom
w = 1.5; %angstrom
kBZ = 1.0472; %angstrom^(-1)
hbar = 6.582119E-16; %eV s
me = 510999*(1E-10)^2/(9E16); %eV A^2

%%
kh = sqrt((2*me*Eh)./(hbar^2));
kph = sqrt((2*me*(Eh+V0))./(hbar^2));

k = sqrt((2*me*E)./(hbar^2));
kp = sqrt((2*me*(E+V0))./(hbar^2));

C1 = cosh(k.*(a-w)./1j).*cos(kp.*w);
C2 = (1/2)*((k./(1j*kp))-(kp*1j./k)).*sinh(k.*(a-w)./1j).*sin(kp.*w);
K= acos(C1+C2)./a;

C1h = cosh(kh.*(a-w)./1j).*cos(kph.*w);
C2h = (1/2)*((kh./(1j*kph))-(kph*1j./kh)).*sinh(kh.*(a-w)./1j).*sin(kph.*w);
Kh= acos(C1h+C2h)./a;


   
%%

Kminussquareh = (Kh - kBZ).^2;
subplot(1,2,1)
plot(Kminussquareh,Eh,'k')

xlabel( '$(K - k_{BZ})^2~~ $(\AA$^{-2}$)', 'Interpreter','LaTex');
ylabel('E (eV)');
title('Top of first band')
set(gca,'FontSize',15)

%slope_adjusth = -5.1;
%E0_adjusth = -3.05943;
slope_adjusth = 1;
E0_adjusth = 0;

fith = polyfit(Kminussquareh,Eh,1);
slopeh = fith(1)
E_0top = fith(2) + E0_adjusth
effective_massh = hbar^2/(2*slopeh)
mass_ratioh = effective_massh / me


hold off


%%
Kminussquare = (K - kBZ).^2;
subplot(1,2,2)
plot(Kminussquare,E,'k')
xlabel( '$(K - k_{BZ})^2~~ $(\AA$^{-2}$)', 'Interpreter','LaTex');
ylabel('E (eV)');
title('Bottom of second band')
set(gca,'FontSize',15)
box on

fit = polyfit(Kminussquare,E,1);
slope = fit(1)
E_0bottom = fit(2)
effective_mass = hbar^2/(2*slope)
mass_ratio = effective_mass / me

hold off