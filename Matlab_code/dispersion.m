clear all

E = -3:.00001:40; %eV

V0 = 1; %eV
a = 3; %angstrom
w = 1.5; %angstrom
kBZ = 1.0472; %angstrom^(-1)
hbar = 6.582119E-16; %eV s
me = 510999*(1E-10)^2/(9E16); %eV A^2


%%
k = sqrt((2*me*E)./(hbar^2));
kp = sqrt((2*me*(E+V0))./(hbar^2));

K00 = acos(cos(k.*a))./a; %for V0=0

%C1 = cos(k.*(a-w)).*cos(kp.*w);
%C2 = (1/2)*(((k./(kp))+(kp./k))).*sin(k.*(a-w)).*sin(kp.*w);
%K= acos(C1 - C2)./a;

C1 = cosh(k.*(a-w)./1j).*cos(kp.*w);
C2 = (1/2)*((k./(1j*kp))-(kp*1j./k)).*sinh(k.*(a-w)./1j).*sin(kp.*w);
K= acos(C1+C2)./a;
   

%% 
subplot(1,2,1)
plot(K00,E,'k')
%scatter(K,E,1,[0 0 0])
hold on
xlabel('K (Å^{-1})');
ylabel('E (eV)');
set(gca,'FontSize',15)
plot(-K00,E,'k')
axis([-1.05 1.05 0 40]);
title('V_0 = 0 eV');
%%


subplot(1,2,2)
plot(K,E,'k')
hold on
plot(-K,E,'k')
%scatter(-K,E,1, [0 0 0])
xlabel('K (Å^{-1})');
ylabel('E (eV)');
set(gca,'FontSize',15)
box on;
axis([-1.05 1.05 -1 40]);
title('V_0 = 1 eV')
%title('V_0 = 1 eV', 'Interpreter','LaTex');
hold off
