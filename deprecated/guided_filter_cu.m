function q = guided_filter_gpu(guide, target, radius, eps)

[g, h, w] = toGPUArray(guide);
t = toGPUArray(target);
gq = cuGuidedFilter(g, t, radius, eps, h);
q = fromGPUArray(gq, w, h);

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