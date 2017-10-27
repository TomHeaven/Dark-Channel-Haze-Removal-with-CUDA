% Dark-Channel-Haze-Removal-CUDA
% Originally authored by Kaiming He
% Optimized with CUDA by TomHeaven, hanlin_tan@nudt.edu.cn, 2016.12.02

warning('off','all');

% Config
filepath = '/Users/tomheaven/Documents/dehaze_data';
fileext = 'png';
useGPU = true;
omega = 0.55;
win_size = 3;
savepath = filepath;

files = dir([filepath, sprintf('/*.%s',fileext)]);

for i = 1 : length(files)
    fprintf('Processing image %d\n', i);
    filename = files(i).name;
    image = double(imread(sprintf('%s/%s', filepath, filename)))/255;  
    result = dehaze_fast(image, omega, win_size, useGPU);
    imwrite(result, sprintf('%s/dh_%s', savepath, filename)); 
end


warning('on','all');