function [ radiance ] = dehaze_fast(image, omega, win_size, useGPU )
%DEHZE Summary of this function goes here
%   Detailed explanation goes here

if ~exist('omega', 'var')
    omega = 0.95;
end

if ~exist('win_size', 'var')
    win_size = 15;
end

r = 15;
res = 0.001;

[m, n, ~] = size(image);

start = tic;

if useGPU
    dark_channel = get_dark_channel_gpu(image, win_size); % 0.03 s
else
    dark_channel = get_dark_channel(image, win_size); % 11.12 s
end

fprintf('tm1 = %f\n', toc(start)); start = tic;

atmosphere = get_atmosphere(image, dark_channel); % 0.27s
clear dark_channel;

fprintf('tm2 = %f\n', toc(start)); start = tic;

if useGPU
    image = gpuArray(image);
end

trans_est = get_transmission_estimate(image, atmosphere, omega, win_size, useGPU); %CPU: 10.98 s GPU 0.34 s

fprintf('tm3 = %f\n', toc(start)); start = tic;

x = guided_filter(rgb2gray(image), trans_est, r, res, useGPU);  %1.75 s

fprintf('tm4 = %f\n', toc(start)); 

transmission = reshape(x, m, n);

radiance = get_radiance(image, transmission, atmosphere);   % 0.00 s

if useGPU
    radiance = gather(radiance);
end

end

