clear all
k = 8.6173E-5; %eV/K
Z = 12;
J = 0.01; %eV
Tc = (Z*J)/(4*k)

for t = 1:5000 % t = 1:500 % T(t) = t; %  x = .01:.01:5;
   T(t) = .08*t;
   
x = .001:.001:5;

Szz = (1/2)*tanh((Z*J*x)./(2*k.*T(t)));

diff = abs(x-Szz);
[g(t), Sz(t)] = min(diff);
end
    
plot(T,.001*Sz,'k')
ylabel('< S_z>')
xlabel('T (K)')
axis([0 400 0 .6])
set(gca,'FontSize',13)