

h1 = gscatter(dRrec,tsen,classes,'krb','ov^',[],'off');
h1(1).LineWidth = 2;
h1(2).LineWidth = 2;
h1(3).LineWidth = 2;
%h1(4).LineWidth = 2;
legend('H2O dark','H2O UV','O2 dark','O2 UV','best')
hold on


X = [RRecovery,tau_res];
MdlLinear = fitcdiscr(X,response);


MdlLinear.ClassNames([1 2])
K = MdlLinear.Coeffs(1,2).Const;
L = MdlLinear.Coeffs(1,2).Linear;


f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f,[.9 7.1 0 2.5]);
h2.Color = 'r';
h2.LineWidth = 2;

hold off




