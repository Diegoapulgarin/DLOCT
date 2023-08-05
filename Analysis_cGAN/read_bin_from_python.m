
% path = 'D:\DLOCT\TomogramsDataAcquisition\ExperimentalTomogram';
% file_imag = strcat( path,'\ExperimentalROI_corrected5_DL_resampled_of_imag.bin');
% file_real = strcat( path,'\ExperimentalROI_corrected5_DL_resampled_of_real.bin');
% dims = [350, 384, 384];
% fid = fopen(file_imag,'r','b');
% binfile_imag = fread(fid,prod(dims),'double');
% fid2 = fopen(file_real,'r','b');
% binfile_real = fread(fid2,prod(dims),'double');
% reshaped_imag = reshape(binfile_imag,dims);
% reshaped_real = reshape(binfile_real,dims);
%%
clc , clear all
path = 'D:\DLOCT\TomogramsDataAcquisition\ExperimentalTomogram';
% Set the path to the .mat file
file_imag = strcat( path,'\ExperimentalROI_corrected5_DL_resampled_of_imag.mat');
file_real = strcat( path,'\ExperimentalROI_corrected5_DL_resampled_of_real.mat');

% Load the .mat file
data_struct_imag = load(file_imag);
data_struct_real = load(file_real);

% Access the imag_data variable
imag_data = data_struct_imag.imag_data;
real_data = data_struct_real.real_data;

%%
plot = squeeze(10*log(abs(real_data(256,:,:)+1i*imag_data(256,:,:)).^2));
imagesc(plot)
colormap gray