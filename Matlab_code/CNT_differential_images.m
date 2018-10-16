
%google matlab imshowpair

%imref2d for establishing spatial references

%%
%{
ps03 = imread('ps03.jpg');
ps20 = imread('ps20.jpg');
ps35 = imread('ps35.jpg');
ps40 = imread('ps40.jpg');
ps47 = imread('ps47.jpg');
ps56 = imread('ps56.jpg');
ps70 = imread('ps70.jpg');
ps86 = imread('ps86.jpg');

%}
%% diff plots

%%{
subplot(3,3,1)
imshowpair(ps03,ps20,'diff');
title('3% - 20%')
hold on
subplot(3,3,2)
imshowpair(ps03,ps35,'diff');
title('3% - 35%')

subplot(3,3,3)
imshowpair(ps03,ps40,'diff');
title('3% - 40%')

subplot(3,3,4)
imshowpair(ps03,ps47,'diff');
title('3% - 47%')

subplot(3,3,5)
imshowpair(ps03,ps56,'diff');
title('3% - 56%')

subplot(3,3,6)
imshowpair(ps03,ps70,'diff');
title('3% - 70%')

subplot(3,3,7)
imshowpair(ps03,ps86,'diff');
title('3% - 86%')
%}


%%

% Magnitude of mean changes in images
%%{
step0 = mean(mean(mean(ps03-ps03)));
step1 = mean(mean(mean(ps20-ps03)));
step2 = mean(mean(mean(ps35-ps03)));
step3 = mean(mean(mean(ps40-ps03)));
step4 = mean(mean(mean(ps47-ps03)));
step5 = mean(mean(mean(ps56-ps03)));
step6 = mean(mean(mean(ps70-ps03)));
step7 = mean(mean(mean(ps86-ps03)));
%}

% Plot Absolute intensities
%{
step0 = mean(mean(mean(ps03)));
step1 = mean(mean(mean(ps20)));
step2 = mean(mean(mean(ps35)));
step3 = mean(mean(mean(ps40)));
step4 = mean(mean(mean(ps47)));
step5 = mean(mean(mean(ps56)));
step6 = mean(mean(mean(ps70)));
step7 = mean(mean(mean(ps86)));
%}

steps = [step0 step1 step2 step3 step4 step5 step6 step7];
RH = [03 20 35 40 47 56 70 86];

subplot(3,3,8)
scatter(RH, steps)
% Avg intensity plot)

xlabel('RH (%)')
ylabel('Avg. intensity (arb. units)')

%% HISTOGRAM DATA
%%{
h03 = imhist(ps03(:,:,3));
h47 = imhist(ps47(:,:,3));
h86 = imhist(ps86(:,:,3));

h03avg = mean(h03);
h47avg = mean(h47);
h86avg = mean(h86);

%plot(imhist(ps03(:,:,3))

%}
%%
% Plot top line profile
%%{
subplot(3,3,9)
plot(ps03(1,:,1),'k')
hold on
plot(ps40(1,:,1),'b')
plot(ps86(1,:,1),'r')
legend('3% RH', '47% RH', '86% RH')
axis([0, 225, 0, 1500])
ylabel('Number of counts')
xlabel('Pixel intensity (arb. units)')

%}
%{
%%
% Contours of each .jpg layer
%%{
a = ps03(:,:,1);
b = ps03(:,:,2);
c = ps03(:,:,3);


subplot(2,2,1)
contourf(a)
subplot(2,2,2)
contourf(b)
subplot(2,2,3)
contourf(c)
subplot(2,2,4)
contourf(a+b+c)


%}
%%
hold off