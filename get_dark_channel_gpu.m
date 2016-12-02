function dark_channel = get_dark_channel_gpu(image, win_size)

R = image(:,:,1);
G = image(:,:,2);
B = image(:,:,3);

[r, x_height, x_width] = toGPUArray(R);
g = toGPUArray(G);
b = toGPUArray(B);
% execute mex CUDA function
g_dc = cuGetDarkChannel(r, g, b, x_height, win_size);
% fetch result
dark_channel = fromGPUArray(g_dc, x_width, x_height);

end % end function

function [gX, x_height, x_width]  = toGPUArray(X)
%% transform to GPU Array
[x_height, x_width] = size(X);
vX = reshape(X', [1, x_height*x_width])';
gX = gpuArray(vX);
end

function Y = fromGPUArray(gY, x_width, x_height)
Y = gather( reshape(gY, [x_width, x_height])');
end