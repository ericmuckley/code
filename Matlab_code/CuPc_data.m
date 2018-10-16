%%plot pressure
subplot(2,2,1)
plot(timemin, pressureTorr, 'k')
ylabel('Pressure (Torr)')
xlabel('Time (min)')
axis([0 1100 0 25])
hold on

%% plot motional resistance

subplot(2,2,2)
plot(timemin, QCMR_25nm_RTOhm./QCMR_25nm_RTOhm(1))
hold on
plot(timemin, QCMR_25nm_150COhm./QCMR_25nm_150COhm(1))
plot(timemin, QCMR_25nm_250COhm./QCMR_25nm_250COhm(1))
plot(timemin, QCMR_100nm_RTOhm./QCMR_100nm_RTOhm(1))
plot(timemin, QCMR_100nm_150COhm./QCMR_100nm_150COhm(1))
plot(timemin, QCMR_100nm_250COhm./QCMR_100nm_250COhm(1))
plot(timemin, QCMR_emptyOhm./QCMR_emptyOhm(1))

legend('25 nm RT','25 nm 150C', '25 nm 250C ', '100 nm RT','100 nm 150C', '100 nm 250C', 'empty crystal')
ylabel('Motional Resistance (R_m/R_m_0) ')
xlabel('Time (min)')
axis([0 1100 1 3.5])
hold off

%% plot electrical resistance

subplot(2,2,3)
plot(timemin, resistance_25nm_RTOhm./resistance_25nm_RTOhm(1))
hold on
plot(timemin, resistance_25nm_150COhm./resistance_25nm_150COhm(1) + 0.5)
plot(timemin, resistance_25nm_250COhm./resistance_25nm_250COhm(1))
plot(timemin, resistance_100nm_RTOhm./resistance_100nm_RTOhm(1))
plot(timemin, resistance_100nm_150COhm./resistance_100nm_150COhm(1))
plot(timemin, resistance_100nm_250COhm./resistance_100nm_250COhm(1))

legend('25 nm RT','25 nm 150C','25 nm 250C ',  '100 nm RT','100 nm 150C', '100 nm 250C')
ylabel('Electrical Resistance (R/R_0)')
xlabel('Time (min)')
axis([0 1100 .6 1.8])
hold off

%% plot mass

subplot(2,2,4)
plot(timemin, QCMmass_25nm_RTug)
hold on
plot(timemin, QCMmass_25nm_150Cug)
plot(timemin, QCMmass_25nm_250Cug)
plot(timemin, QCMmass_100nm_RTug)
plot(timemin, QCMmass_100nm_150Cug - 0.65 - .0001.*timemin)
plot(timemin, QCMmass_100nm_250Cug + 0.2)
plot(timemin, QCMmass_emptyug + 0.05)

legend('25 nm RT','25 nm 150C','25 nm 250C ',  '100 nm RT' ,'100 nm 150C', '100 nm 250C','empty crystal')
ylabel('Mass gain (\mug)')
xlabel('Time (min)')
axis([0 1100 -.05 .2])
hold off

    