[x,y,z] = peaks(10);
 [xnew, ynew] = meshgrid(linspace(-3,3,100));
 znew = interp2(x,y,z,xnew,ynew, 'spline');
 subplot(2,1,1);
 hold on;
 scatter(x(:), y(:), [], z(:), 'filled');
 contour(x,y,z, -6:7);
 subplot(2,1,2);
 hold on;
 scatter(x(:), y(:), [], z(:), 'filled');
 contour(xnew, ynew, znew, -6:7);