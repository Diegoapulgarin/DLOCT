
path = 'D:\DLOCT\TomogramsDataAcquisition\Fovea\No_motion_corrected';
file1 = 'tomDataOver_Fovea_pol1.mat';
data = fullfile(path, file1);
tom = load(data);
% tom = tom.tomOver;

%%

tomIntFilename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tom_z=(295..880)_x=(65..960)_y=(1..960)__original_pol1_real.bin';
fId1 = fopen(fullfile(path, tomIntFilename), 'w'); % Open
fwrite(fId1, tom(:,:,:,1), 'single'); % Write
fclose(fId1); % Close


tomIntFilename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tom_z=(295..880)_x=(65..960)_y=(1..960)__original_pol1_imag.bin';
fId2 = fopen(fullfile(path, tomIntFilename), 'w'); % Open
fwrite(fId2, tom(:,:,:,2), 'single'); % Write
fclose(fId2); % Close
disp('ok')
%%
z = 250;
plot = squeeze(10*log(abs(tom(:,:,z,1)+1i*tom(:,:,z,2)).^2));
imagesc(plot);
colormap gray