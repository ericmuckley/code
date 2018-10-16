
%% Define sample masses (mg)

C60QCMmass = 8.9289;
C60IDEmass = 13.7333;
sSWNTQCMmass = 0.0955;
sSWNTIDEmass = 0.0305;
mSWNTQCMmass = 0.1093;
mSWNTIDEmass  = 0.0459;
MWNTQCMmass = 0.0802;
MWNTIDEmass = 0.0344;

%% Normalize to sample masses



%% Pressure plot

subplot(2,3,1)
plot(timeminutes, pressure)
xlabel('Time (min)')
ylabel('Pressure (Torr)')

%% Resistance plot UV
subplot(2,3,2)
hold on
plot(timeminutes, sSWNTrUV./sSWNTrUV(1))

plot(timeminutes, mSWNTrUV./mSWNTrUV(1))
plot(timeminutes, MWNTrUV./MWNTrUV(1))
%plot(timeminutes, C60rUV./C60rUV(1))
xlabel('Time (min)')
ylabel('UV Resistance (R/R_0)')

legend('s-SWNT', 'm-SWNT', 'MWNT')
hold off

%% Resistance plot dark
subplot(2,3,3)

hold on
plot(timeminutes, sSWNTr./sSWNTr(1))
plot(timeminutes, mSWNTr./mSWNTr(1))
plot(timeminutes, MWNTr./MWNTr(1))
%plot(timeminutes, C60r./C60r(1))
xlabel('Time (min)')
ylabel('Dark Resistance (R/R_0)')

legend('s-SWNT', 'm-SWNT', 'MWNT')
hold off

%% Mass plot UV

subplot(2,3,5)
plot(timeminutes, sSWNTmUV ./ sSWNTmUV(1))
hold on
plot(timeminutes, mSWNTmUV ./ mSWNTmUV(1))
plot(timeminutes, MWNTmUV ./ MWNTmUV(1))
%plot(timeminutes, C60mUV ./ C60mUV(1))
xlabel('Time (min)')
ylabel('UV Mass change (M/M_0)')

legend('s-SWNT', 'm-SWNT', 'MWNT')
hold off

%% Mass plot

subplot(2,3,6)
plot(timeminutes, sSWNTm ./ sSWNTm(1))
hold on
plot(timeminutes, mSWNTm ./ mSWNTm(1))
plot(timeminutes, MWNTm ./ MWNTm(1))
%plot(timeminutes, C60m ./ C60m(1))
xlabel('Time (min)')
ylabel('Dark Mass change (M/M_0)')

legend('s-SWNT', 'm-SWNT', 'MWNT')
hold off