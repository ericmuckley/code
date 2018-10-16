%% classification learner
mindex = 6;
mindex0 = round(mindex/2); %CUT OUT FIRST 2 DATA POINTS
time = [at(mindex) bt(mindex) kt(mindex)];
mean(time)


aall = [a1(mindex0:mindex) a3(mindex0:mindex) a5(mindex0:mindex) a7(mindex0:mindex) a9(mindex0:mindex) a11(mindex0:mindex)];
ball = [b1(mindex0:mindex) b3(mindex0:mindex) b5(mindex0:mindex) b7(mindex0:mindex) b9(mindex0:mindex) b11(mindex0:mindex)];
kall = [k1(mindex0:mindex) k3(mindex0:mindex) k5(mindex0:mindex) k7(mindex0:mindex) k9(mindex0:mindex) k11(mindex0:mindex)];
minutesall = vertcat(at(mindex0:mindex), bt(mindex0:mindex), kt(mindex0:mindex));


frequencies = vertcat(aall, ball, kall);
classes = vertcat(ones(mindex-mindex0+1,1) - 1, ones(mindex-mindex0+1,1), ones(mindex-mindex0+1,1)+1) ;



all = [frequencies minutesall classes];

%{
%% classification learner SHORT 20 min
mindex = 29; %index to stop minutes at
minutesshort = minutes(1:mindex);


aallshort = [a1(1:mindex) a3(1:mindex) a5(1:mindex) a7(1:mindex) a9(1:mindex) a11(1:mindex)];
ballshort = [b1(1:mindex) b3(1:mindex) b5(1:mindex) b7(1:mindex) b9(1:mindex) b11(1:mindex)];
kallshort = [k1(1:mindex) k3(1:mindex) k5(1:mindex) k7(1:mindex) k9(1:mindex) k11(1:mindex)];
minutesallshort = vertcat(minutesshort, minutesshort, minutesshort);


frequenciesshort = vertcat(aallshort, ballshort, kallshort);
classesshort = vertcat(ones(length(aallshort),1) - 1, ones(length(aallshort),1), ones(length(aallshort),1)+1);

allshort = [frequenciesshort minutesallshort classesshort];



%% nn classifier

nninputs = [frequencies minutesall];
zz = zeros(length(a1),1);

class1 = vertcat(zz+1,zz,zz);
class2 = vertcat(zz,zz+1,zz);
class3 = vertcat(zz,zz,zz+1);
nnclasses = [class1 class2 class3];

nninputsshort = [frequenciesshort minutesallshort];
zzs = zeros(length(a1(1:mindex)),1);


class1s = vertcat(zzs+1,zzs,zzs);
class2s = vertcat(zzs,zzs+1,zzs);
class3s = vertcat(zzs,zzs,zzs+1);
nnclassesshort = [class1s class2s class3s];
%}