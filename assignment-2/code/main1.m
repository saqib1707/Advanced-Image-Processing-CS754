q=1;
img = imread('../data/barbara256.png');
[imgrow,imgcol] = size(img);
noise_var = 10;
noise = randn(size(img));
noise_img = double(img)+noise;
% noise_img = imadd(double(img), noise);
% figure,imshow(img);
% figure,imshow(noise_img,[]);

% img = im2double(imread('../data/barbara256.png'));
% [imgrow,imgcol] = size(img);
% noise_var = 10;
% noise = randn(size(img))*sqrt(noise_var)/255.0;
% noise_img = imadd(img, noise);
% figure,imshow(img);
% figure,imshow(noise_img);
% dct_image = dct2(noise_img);

psize=8;
lambda=10.0;
finalimg = zeros(size(img));
phi = return_phi(q, psize);
count=0;
for i=0:(imgrow/psize)-1
    for j=0:(imgcol/psize)-1
        count=count+1;
        patch = noise_img(i*psize+1:(i+1)*psize,j*psize+1:(j+1)*psize);
        patchvec = patch(:);
        theta = ista(patchvec, phi, lambda);
        finalimg(i*psize+1:(i+1)*psize,j*psize+1:(j+1)*psize) = reshape(phi*theta,size(patch));
        disp(count);
    end
end
imshow(finalimg,[]);
imwrite(uint8(finalimg), 'recimg10.jpg');