%function [Nit] = meet()
%     x = create_random_sparse_vector();
    indices = randperm(100,10);
    x=zeros(100,1);
    x(indices) = normrnd(0,1.0,10,1);
    h = [1 2 3 4 3 2 1]/16;
    mag_x = norm(x);
    
    y = conv(x,h);
    disp(size(y));
    y=y';
    yn = y + randn(1,106) * 0.01 * mag_x; % Add 1% noise

    yn = yn';
    x = x';
    
    H = convmtx(h',100); % create circulant matrix of kernel

    lambda = 50;
    lambda_cap = lambda*2*(0.01*mag_x)^2;
    alpha = max(eig(H'*H));
    
    % % returns deblurred signal and no. iterations to converge
    % [x_deblurred,Nit] = ista(yn,H,lambda_cap,alpha,zeros(size(x)), 10^-5);

    figure;
    stem(yn);
    title('Noisy signal');

    figure;
    subplot(1,2,1);
    stem(x);
    title('Original signal');
    
    % subplot(1,2,2);
    % stem(x_deblurred);
    % title('Deblurred signal');
    
%end