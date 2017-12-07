function trans_est = get_transmission_estimate(image, atmosphere, omega, win_size, useGPU)

[m, n, ~] = size(image);

rep_atmosphere = repmat(reshape(atmosphere, [1, 1, 3]), m, n);


if useGPU
    dark_channel = get_dark_channel_gpu( image ./ rep_atmosphere, win_size); 
else
    dark_channel = get_dark_channel( image ./ rep_atmosphere, win_size);
end


trans_est = 1 - omega * dark_channel;

end