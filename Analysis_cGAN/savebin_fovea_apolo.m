
path = 'D:\DLOCT\TomogramsDataAcquisition\Fovea\No_motion_corrected';
file1 = 'tomDataOver_Fovea_pol1_Overlap.mat';
file2 = 'tomDataOver_Fovea_pol2_Overlap.mat';
data1 = fullfile(path, file1);
data2 = fullfile(path, file2);
tom1 = load(data1);
tom2 = load(data2);
tom1 = tom1.tomDataOver;
tom2 = tom2.tomDataOver;
%%
tom = zeros(586,896,960,2);
tom(:,:,:,1) = tom1;
tom(:,:,:,2) = tom2;
%%
tomint = abs(tom).^2;
%%
path = 'C:\Users\labfisica\Documents\OCT_Advanced_Postprocessing\Data\DLOCT\RetinalImaging\Fovea\[p.SHARP][s.Eye2a][10-09-2019_13-14-42]';
tomIntFilename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..960)_y=(1..960)_reconstructed_Overlap.bin';
fId1 = fopen(fullfile(path, tomIntFilename), 'w'); % Open
fwrite(fId1, tomint, 'single'); % Write
fclose(fId1); % Close

disp('ok')
%%
z = 500;
plot = squeeze(10*log(abs(tom(:,:,z,1)).^2));
imagesc(plot);
colormap gray