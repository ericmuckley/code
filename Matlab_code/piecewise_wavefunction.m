%% water thickness part



kBT = .0256;
v = 30; %Angstrom^3
H = -.4;
p0 = 25.8;


p = 0:.01:p0; %pressure range 

d = v*((H)./(6*pi*kBT.*log(p/p0))).^(1/3); %angstrom


dnorm = 0.2*d ;


plot(p,dnorm)
hold on
plot(H2O_pressure_torr, resistance_ohm)


xlabel('H_2O Pressure (Torr)')
ylabel('R_m (Angstrom)')





%% plot piecewise

% get some data
     %x=-1.5:.00001:1.5;
     psi=nan*ones(size(p));
% compute each epoch
    psi(p<0.75)=  p(p<0.75);
     
    
     psi(0.75<=p&p<=0.85)=(p(0.75<=p&p<=0.85));
     
  
    psi(0.85<p)= p(0.85<p);
    
    plot(p,psi)
     
     
     %% plot
     
     plot(,psi,'k')
     hold on
     plot(x,density,'--')
     legend('\Psi (x)', '\Psi(x)^*\Psi(x)')
     set(gca,'FontSize',12);

     
     hold off
     