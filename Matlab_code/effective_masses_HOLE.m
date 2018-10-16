clear all

E = .0141:.0001:.01425; %eV

V0 = 1; %eV
a = 3; %angstrom
w = 1.5; %angstrom
kBZ = 1.0472; %angstrom^(-1)
hbar = 6.582119E-16; %eV s
me = 9.10938E-31; %kg

%%
k = sqrt((2*me*E)./(hbar^2));
kp = sqrt((2*me*(E+V0))./(hbar^2));

K00 = acos(cos(k.*a))./a; %for V0=0

C1 = cos(k.*(a-w)).*cos(kp.*w);
C2 = (1/2)*(((k./(kp))+(kp./k))).*sin(k.*(a-w)).*sin(kp.*w);
K= acos(C1 - C2)./a;
 
%%

Kminussquare = (K - kBZ).^2;
plot(Kminussquare,E,'k')
xlabel( '$(K - k_{BZ})^2~~ $(\AA$^{-2}$)', 'Interpreter','LaTex');
ylabel('E (eV)');
title('Slope at bottom of second band')
set(gca,'FontSize',15)
fit = polyfit(Kminussquare,E,1)

fithplot = polyval(fit,E);
hold on
plot(Kminussquare, fithplot)
legend('data','fit')

slope = fit(1)
E_0top = fit(2)

effective_mass = hbar^2/(2*slope)

mass_ratio = effective_mass / me

