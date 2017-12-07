function sum_img = window_sum_filter_gpu(image, r)

% sum_img(x, y) = = sum(sum(image(x-r:x+r, y-r:y+r)));

[h, w] = size(image);
vImage = toGPUArray(image);
s = cuWindowSumFilter(vImage, h, r);
sum_img = fromGPUArray(s, w, h);

end

function [gX, x_height, x_width]  = toGPUArray(X)
%% transform to GPU Array
[x_height, x_width] = size(X);
vX = reshape(X', [1, x_height*x_width])';
gX = gpuArray(vX);
end

function Y = fromGPUArray(gY, x_width, x_height)
Y = gather( reshape(gY, [x_width, x_height])');
end

