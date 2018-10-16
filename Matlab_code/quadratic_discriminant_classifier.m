
load fisheriris %load data

%identify classifying variables
PL = meas(:,3); 
PW = meas(:,4);



%%

h1 = gscatter(PL,PW,species,'krb','ov^',[],'off'); %scatter plot variables
h1(1).LineWidth = 2; %make lines wider
h1(2).LineWidth = 2;
h1(3).LineWidth = 2;
legend('Setosa','Versicolor','Virginica','Location','best')
hold on

X = [PL,PW]; 
MdlLinear = fitcdiscr(X,species); %create classifier

%get coefficents from lines which separate 2nd and 3rd classes
MdlLinear.ClassNames([2 3])
K = MdlLinear.Coeffs(2,3).Const;
L = MdlLinear.Coeffs(2,3).Linear;

%plot curve that separates 2nd and 3rd classes
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f,[.9 7.1 0 2.5]);
h2.Color = 'r';
h2.LineWidth = 2;

%get coefficents from lines which separate 1st and 2nd classes
MdlLinear.ClassNames([1 2])
K = MdlLinear.Coeffs(1,2).Const;
L = MdlLinear.Coeffs(1,2).Linear;

%plot curve that separates 1st and 2nd classes
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h3 = ezplot(f,[.9 7.1 0 2.5]);
h3.Color = 'k';
h3.LineWidth = 2;
axis([.9 7.1 0 2.5])
xlabel('Petal Length')
ylabel('Petal Width')
title('{\bf Linear Classification with Fisher Training Data}')


hold off


