
path = 'C:\Users\USER\Documents\GitHub\Fovea';
file1 = 'tomDataOverpol0.mat';
file2 = 'tomDataOverpol1.mat';
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
path = 'C:\Users\USER\Documents\GitHub\Fovea';
tomIntFilename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..960)_y=(1..960)_reconstructed.bin';
fId1 = fopen(fullfile(path, tomIntFilename), 'w'); % Open
fwrite(fId1, tomint, 'single'); % Write
fclose(fId1); % Close

disp('ok')
%%
z = 500;
plot = squeeze(10*log(abs(tom(:,:,z,1)).^2));
imagesc(plot);
colormap gray