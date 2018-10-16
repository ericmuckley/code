close all
clear all

%solar panel breakdown:
%type =  [efficiency, $/m^2]
%thin film CdTe - DM 77W CdTe Solar Module from DM Solar
tf = [.12 64/.72];
%monocrystalline Si - Hyundai 270w Mono Solar Module from DM solar
msi = [.185 243/1.617];
%polycrystalline Si - REC 315 PE-72 Solar Panel from Beyond Oil Solar
psi = [.159 292/1.95];


area = 0 : 100 : 90000; %m^2


%if we use 4.3 kWh/m^2/day, then an average TN day gives 197 W/m^2.

%max power (MW)
MP_tf = 197*area.*tf(1)/1000000;
MP_msi = 197*area.*msi(1)/1000000;
MP_psi = 197*area.*psi(1)/1000000;

%cost (millions of $)
cost_tf = area.*tf(2)./1000000;
cost_msi = area.*msi(2)./1000000;
cost_psi = area.*psi(2)./1000000;


subplot(2,1,1)
plot(area, MP_msi,'k')
hold on
plot(area, MP_psi)
plot(area, MP_tf)
legend('Monocrystalline Si','Polycrystalline Si','Thin Film CdTe')
box on
xlabel('Area Utilized (m^2)')
ylabel('Power Output (MW)')
set(gca,'FontSize',12)
title('Potential Power Generation at ORNL by Photvoltaics')
axis([0 50000 0 2])

subplot(2,1,2)
plot(cost_msi, MP_msi, 'k',cost_psi, MP_psi, cost_tf,  MP_tf )
legend('Monocrystalline Si','Polycrystalline Si','Thin Film CdTe')
box on
xlabel('Cost (millions of $)')
ylabel('Power Output (MW)')
set(gca,'FontSize',12)
axis([0 8 0 2.3])
