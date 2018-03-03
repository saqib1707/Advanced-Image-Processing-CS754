indices=randperm(100,10);
x=zeros(100,1);
x(indices) = normrnd(0,1.0,10,1);
h = ([1,2,3,4,3,2,1]/16)';
% x = [4,5,6,1]';
% h=[3,2,1]';
xmag = norm(x);
c = conv(x,h);
noise = normrnd(0,0.05*xmag,size(c));
y = c+noise;

phi = zeros(size(y,1),size(x,1));
for i = 1:size(x,1)
    phi(i:i+size(h,1)-1,i) = h;
end
lambda = 1.0;
theta = ista(y, phi, lambda);
recerror  = norm(theta - x);
disp(recerror);