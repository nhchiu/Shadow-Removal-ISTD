function [corrected_im,param] =color_adjustment(shadow_free, shadow,shadow_mask)

linear = cell(3,1);
shadow_mask =repmat(shadow_mask,[1,1,3]);

% Select only non-shadow pixels on the shadow-free image (source)
%   and the shadow image (target)

source = double(shadow_free(shadow_mask==0))/255;
target =  double(shadow(shadow_mask==0))/255;

% Seperate each color channel
source = reshape(source,[],3);
target = reshape(target,[],3);

% Fit a linear regression for each color channel.
for i = 1:3
    linear{i} = regress(target(:,i),[ones(size(source,1),1) source(:,i)]);
end
param= [linear{1}' linear{2}' linear{3}'];

% Recover the shadow-free image using the linear regressions above
corrected_im = double(shadow_free)/255;
corrected_im(:,:,1) = corrected_im(:,:,1)*param(2) + param(1);
corrected_im(:,:,2) = corrected_im(:,:,2)*param(4) + param(3);
corrected_im(:,:,3) = corrected_im(:,:,3)*param(6) + param(5);
corrected_im = uint8(corrected_im*255);
end





