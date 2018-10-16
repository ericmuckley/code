clear all

%% water thickness part
%water layer thickness in absence of applied bias on a solid salt
%planar double layer model

A = -0.4 %eV
kBT = .0256; %eV
v = 30; %Angstrom^3
p0 = 23.8; %Torr


p = 0:.01:23.8;

d = v*(A./(6*pi*kBT.*log(p/p0))).^(1/3); %in Angstroms
plot(p,d)
xlabel('H_2O Pressure (Torr)')
ylabel('H_2O thickness (Angstrom)')


%% COMPUTE PIECEWISE
% get some data
     x=0:.00001:26;
     psi=nan*ones(size(x));
% compute each epoch
    psi(x<-0.75)= x(x<-0.75);
     %psi(x<-0.75)= A_1* cos(k* x(x<-0.75));
    
     psi(-0.75<=x&x<=0.75)=((sin((x(-0.75<=x&x<=0.75)))));
     
     %psi(0.75<x)= 2*real( A_3) * cos(k*x(0.75<x));
    psi(0.75<x)= x(0.75<x);
     
     
     
     %% PLOT
    
     plot(x,psi,'k')
     hold on
     plot(x,psi+1,'--')
     legend('R_m', 'Fit')
     xlabel('Position (\AA)', 'interpreter', 'latex')
     ylabel('Amplitude')
     set(gca,'FontSize',12);
    title('Wavefunction at bottom of second band')
     % title('Wavefunction at top of first band')
     box on
     
     hold off
     