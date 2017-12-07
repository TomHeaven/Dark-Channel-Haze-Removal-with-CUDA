% Dark-Channel-Haze-Removal-CUDA
% Originally authored by Kaiming He
% Optimized with CUDA by TomHeaven, hanlin_tan@nudt.edu.cn, 2016.12.02

warning('off','all');

% Config
useGPU = true;
omega = 0.95;
win_size = 3;


tic;
image = double(imread('forest.jpg'))/255;

image = imresize(image, 2.5);

result = dehaze_fast(image, omega, win_size, useGPU);
toc;

% figure, imshow(image)
figure, imshow(result)

warning('on','all');