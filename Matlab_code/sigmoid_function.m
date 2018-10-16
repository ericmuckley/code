x=-5:.1:5;
s = 1./(1+exp(-x));
plot(x,s,'color','k','linewidth', 4)
yL = ylim;
line([0 0], yL,'color','k','linewidth',2) %y-axis
set(gca,'FontSize',25)
box off