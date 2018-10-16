clear all
D = .005; %eV
Ji1 = .010; %eV
Ji2 = -.004; %eV
d = 3; %angstrom
limit = pi/d;

q = -limit:.0001:limit;

Hex = -(Ji1*2 + Ji2*2 + 4*D );
Ha = 2*(1-cos(q*d)) - 2* (1-cos(2*q*d));

E =  Hex + 2* Ha;

plot(q,E,'k')
ylabel('E (eV)')
%xlabel('q ( $\rm{\AA} ^{-1})$ ','interpreter','LaTex');
xlabel('q ( $\pi / d)$ ','interpreter','LaTex');
axis([-limit limit -5 10])
set(gca,'FontSize',14)