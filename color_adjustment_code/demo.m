%%%%%%%%
% ICCV submission #2450
% This script demonstrates our color adjustment method to correct 
%   the color inconsistency between the shadow image and shadow-free image
%   of the ISTD testing set. We provide here an example image from the ISTD
%   testing set ('114-5.png')
% Usage: 
%       matlab demo.m
%%%%%%%% 


shadow = imread('114-5_shadow.png');
shadow_free = imread('114-5_shadow_free_original.png');
shadow_mask = imread('114-5_shadow_mask.png');


[corrected_im,w] = color_adjustment(shadow_free,shadow,shadow_mask);


figure(1); 
imshow([shadow,shadow_free]);
title('shadow image vs original shadow-free image');

figure(2); 
imshow([shadow,corrected_im]);
title('shadow image vs corrected shadow-free image');
