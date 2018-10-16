
%normalize resistances to R/Ro for humidity experiment
%(this experiment starts at data point number 1)
MWNT_R_normalized_H2O = MWNT_resistance_ohm / MWNT_resistance_ohm(1);
SWNT_R_normalized_H2O = SWNT_resistance_ohm / SWNT_resistance_ohm(1);
GNP_R_normalized_H2O = GNP_resistance_ohm / GNP_resistance_ohm(1);

%normalize resistances fto R/Ro for oxygen experiment
%(this experiment starts at data point number 6835)
MWNT_R_normalized_O2 = MWNT_resistance_ohm / MWNT_resistance_ohm(6835);
SWNT_R_normalized_O2 = SWNT_resistance_ohm / SWNT_resistance_ohm(6835);
GNP_R_normalized_O2 = GNP_resistance_ohm / GNP_resistance_ohm(6835);

%normalize mass changes to M/Mo for oxygen experiment
%(this experiment starts at data point number 6835)
MWNT_M_normalized_O2 = MWNT_QCM_mass_ug / MWNT_QCM_mass_ug(6835);
SWNT_M_normalized_O2 = SWNT_QCM_mass_ug / SWNT_QCM_mass_ug(6835);
GNP_M_normalized_O2 = GNP_QCM_mass_ug / GNP_QCM_mass_ug(6835);

%Note: mass is already normalized for the H2O experiment, because 
%this experiment was first

%%
%%%%%%%%%%%%
%RH experiment

%plot R/Ro for RH experiment
subplot(3,2,1)
plot(min_elapsed, MWNT_R_normalized_H2O, min_elapsed, SWNT_R_normalized_H2O, min_elapsed, GNP_R_normalized_H2O,'k')
legend('MWNT', 'SWNT','GNP')
xlabel('Time (min)')
ylabel('R/R_0');
axis([0 450 .95 1.3])
title('Humidity experiments')

%plot mass adsorption for RH experiment
subplot(3,2,3)
plot(min_elapsed, MWNT_QCM_mass_ug, min_elapsed, SWNT_QCM_mass_ug, min_elapsed, GNP_QCM_mass_ug./10, 'k')
legend('MWNT', 'SWNT','GNP')
xlabel('Time (min)')
%ylabel('Mass adsorbed (\mug)');
ylabel('Mass adsorbed (M/M_0)');
axis([0 450 -0.1 .4])
box on

%plot RH 
subplot(3,2,5)
plot(min_elapsed, RH_percent, 'k')
axis([0 450 0 100])
xlabel('Time (min)')
ylabel('Relative Humidity (%)');
box on



%%
%%%%%%%%%%%%%%%%%%%
%O2 experiment

%plot R/Ro for O2 experiment
subplot(3,2,2)
plot(min_elapsed, MWNT_R_normalized_O2, min_elapsed, SWNT_R_normalized_O2, min_elapsed, GNP_R_normalized_O2,'k')
legend('MWNT', 'SWNT','GNP')
xlabel('Time (min)')
ylabel('R/R_0');
axis([450 900 .95 1.4])
title('O_2 experiments')

%plot mass adsorption for O2 experiment
subplot(3,2,4)
plot(min_elapsed, MWNT_M_normalized_O2, min_elapsed, SWNT_M_normalized_O2, min_elapsed, GNP_M_normalized_O2, 'k')
legend('MWNT', 'SWNT','GNP')
xlabel('Time (min)')
%ylabel('Mass adsorbed (\mug)');
ylabel('Mass adsorbed (M/M_0)');
axis([450 900 .4 7])
box on

%plot O2 pressure
subplot(3,2,6)
plot(min_elapsed, O2_pressure_torr, 'k')
axis([450 900 0 25])
xlabel('Time (min)')
ylabel('O_2 Pressure (Torr)');
box on
hold off
%%

