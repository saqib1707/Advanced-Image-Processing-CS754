function [theta] = ista(Y, A, lambda)
[rowY,colY] = size(Y);
[rowA,colA] = size(A);
theta = randn(colA, colY);
e = eig(A.'*A);
alpha = max(e)+1.0;
thr = lambda/(2*alpha);
niter = 50;
% basic objective/loss function
for i=1:niter
    soft1 = theta + (1/alpha)*A.'*(Y-A*theta);
    theta = wthresh(soft1,'s',thr);
%     diff = Y-A*theta;
%     J = norm(diff).^2+lambda*norm(theta,1);
%     loss(i) = J;
end
% x = [1:niter];
% y = loss;
% figure, plot(x,y);
end