%% Folders: 
%- Data: three gray images.
%% m-File:
% The imported image is divided into patches and vectorized into a column vector.
% named patch in the m-file, which is then used as the input vector x in
% the linear system y = Ax.
% The reconstructed patches are then regroup to produce the recovered image.

%% Complete glasso.m first, and pass the test using test_glasso.m.
%% After that, run this demo.

%% functions:
% dct: discrete cosine transform of the vector patch which transforms the signal (patch) in 
% the space domain to the cosine domain, which is often sparse 
%% glasso: the function you need to complete

clear;clc;close all;
addpath('./Data');

%% load image
rng('default')
img = imread('phantom.png');
img = double(img);
% img = double(rgb2gray(img));
img = img./max(img(:));
[D,~] = size(img); % square image

n = 8; % image patch size n*n
M = 25; % number of measurments
img_recon = zeros(size(img));

A = randn(M,n^2);% Generate Sensing Matrix A
lambda = 0.18;

%% image compression and reconstruction
for i = 1:n:D
    for j = 1:n:D
        x_0 = img(i:i+n-1,j:j+n-1); % generate image patch
        x_0 = x_0(:);
        
        %% compress the data
        dct_mtx = dctmtx(length(x_0)); % dct matrix
        y = A*x_0 + 0.01*randn(M,1); 
        
        %% put your glasso method below: alpha = glasso(A, y, dct_mtx, lambda, tol, maxiter)
        alpha = glasso( A, y, dct_mtx, lambda, 1e-8 );
        %% reconstruct the image
        x_hat = reshape(alpha,n,n); % reshape image patch
        img_recon(i:i+n-1,j:j+n-1) = x_hat;
    end
end

%% plot the reconstructed image
figure;
subplot(2,1,1);imshow(img,[]); title('original image');
subplot(2,1,2);imshow(img_recon,[]); title('reconstructed image');

%% disp Peak Signal-noise ratio (PSNR)
Img_Max = max(img(:));
MSE = 1/D^2*sum(sum((img - img_recon).^2));
PSNR = 10*log10(Img_Max^2/MSE);
disp(['Mean square error (MSE)       : ',num2str(MSE)]);
disp(['Peak signal-noise ratio (PSNR): ',num2str(PSNR),' dB']);

