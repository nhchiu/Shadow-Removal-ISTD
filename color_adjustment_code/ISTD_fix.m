clear;
clc;
addpath('../color_adjustment_code/');
root1 = '../train/train_A/';
root2 = '../train/train_B/';
root3 = '../train/train_C/';
save_root = '../train/train_C_fixed_official/';
mkdir(save_root);
image_path1 = dir(fullfile(root1,'*.png'));
for i=1: length(image_path1)
    name1 = image_path1(i).name;
    shadow = double(imread([root1 name1]));
    shadow_mask = imread([root2 name1]);
    shadow_free = double(imread([root3 name1]));
    [corrected_im,w] = color_adjustment(shadow_free,shadow,shadow_mask);
    imwrite(corrected_im,[save_root  image_path1(i).name]);
    %imshow([corrected_im,shadow,shadow_free]);
    %pause
end