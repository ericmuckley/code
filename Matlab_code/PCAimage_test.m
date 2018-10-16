%% Prepare image
clear all

I = (imread('SurfPotCNTs.png')); %read in image
Iflip =  flipdim(I ,1); %flip image vertically
Idata = double(I); %read in dbl data from image

%% Plots
%show image
subplot(3,3,1)
imshow(I)
%imshow('SurfPotCNTs.png')
title('Original image')
hold on 

%separate RGB modes from image
Red = Iflip(:,:,1);
Green = Iflip(:,:,2);
Blue = Iflip(:,:,3);

%plot red
subplot(3,3,2)
contourf(Red)
colormap(hot)
colorbar
title('Red')

%plot green
subplot(3,3,3)
contourf(Green)
colorbar
title('Green')

%plot blue
subplot(3,3,4)
contourf(Blue)
colorbar
title('Blue')

%% PCA

X = reshape(Idata,size(Idata,1)*size(Idata,2),3); %prepare image data for PCA
coeff = pca(X); %perform PCA
PCAimage = coeff; %create matrix of PCA data for total image

%Create PCA images
Itransformed = X*coeff;
Ipc1 = reshape(Itransformed(:,1),size(I,1),size(I,2));
Ipc2 = reshape(Itransformed(:,2),size(I,1),size(I,2));
Ipc3 = reshape(Itransformed(:,3),size(I,1),size(I,2));

%plot PCA images
subplot(3,3,5)
imshow(Ipc1,[]);
title('PCA 1')
subplot(3,3,6)
imshow(Ipc2,[]);
title('PCA 1')
subplot(3,3,7)
imshow(Ipc3,[]);
title('PCA 1')

%Plot sum of all prinicpal components
subplot(3,3,8)
imshow((Ipc1 + Ipc2 + Ipc3),[])
title('Sum of principal components')
colorbar



hold off
