
class0 = ones(2508,1) - 1;
class1 = ones(2508,1);
class2 = ones(2508,1) + 1;
class3 = ones(2508,1) + 2;

classes = vertcat(class0, class1, class2, class3);
press = vertcat(mbar, mbar, mbar, mbar);

fdata = vertcat(arf, co2f, h2of, n2f);
rdata = vertcat(arr, co2r, h2or, n2r);

f_r_class = [fdata rdata classes];
f_r_class_press = [fdata rdata classes press];