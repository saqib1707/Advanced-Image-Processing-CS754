q=2;
img = double(imread('../data/barbara256.png'));
[imgrow,imgcol] = size(img);
psize=8; 
lambda=1.0;
finalimg = zeros(size(img));
countmat = zeros(size(img));
phi = return_phi(q, psize);
psi = return_phi(1, psize);
D = phi*psi;
count=0;
% for i=0:(imgrow/psize)-1
%     for j=0:(imgcol/psize)-1
%         count=count+1;
%         imgpatch = img(i*psize+1:(i+1)*psize,j*psize+1:(j+1)*psize);
%         patchvec = imgpatch(:);
%         patch = phi*patchvec;
%         theta = ista(patch, D, lambda);
%         finalimg(i*psize+1:(i+1)*psize,j*psize+1:(j+1)*psize) = reshape(psi*theta,size(imgpatch));
%         disp(count);
%     end
% end

for i=0:(imgrow-psize)
    for j=0:(imgcol-psize)
        imgpatch = img(i+1:i+psize,j+1:j+psize);
        patchvec = imgpatch(:);
        patch = phi*patchvec;
        theta = ista(patch, D, lambda);
        finalimg(i+1:i+psize,j+1:j+psize) = finalimg(i+1:i+psize,j+1:j+psize)+reshape(psi*theta,size(imgpatch));
        countmat(i+1:i+psize,j+1:j+psize) = countmat(i+1:i+psize,j+1:j+psize) + ones(psize,psize);
    end
end
finalimg = finalimg./countmat;
figure,imshow(img,[]);
figure,imshow(finalimg,[]);