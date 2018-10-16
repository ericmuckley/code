
%input fit parameters, in order Mg, K
%delta R

y = .0;

y0R = [3.13822 4.2066];
x0R = [-0.71494 -0.5009];
A1R = [-3.21708 -4.32665];
t1R = [23.94368 17.38396];

%detection limit using delta R
xR = x0R + t1R.*(-log(-(y0R-y)./A1R))

%input fit parameters, in order Mg, K
%delta M
y0M = [.19181 .5261];
x0M = [-0.83512 -0.20536];
A1M = [-.19735 -.53551];
t1M = [23.44675 13.02556];

%detection limit using delta M
xM = x0M + t1M.*(-log(-(y0M-y)./A1M))



