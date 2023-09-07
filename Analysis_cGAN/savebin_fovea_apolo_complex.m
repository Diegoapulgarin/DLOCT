path = 'C:\Users\USER\Documents\GitHub\Fovea';
tomDownReal = load(fullfile(path,"tomDownReal.mat"));
tomDownReal = tomDownReal.tomDownReal;
tomDownImag = load(fullfile(path,"tomDownImag.mat"));
tomDownImag = tomDownImag.tomDownImag;

%%
tomIntFilename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..512)_y=(1..960)_subsampled_Real.bin';
fId1 = fopen(fullfile(path, tomIntFilename), 'w'); % Open
fwrite(fId1, tomDownReal, 'single'); % Write
fclose(fId1); % Close

tomIntFilename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..512)_y=(1..960)_subsampled_Imag.bin';
fId1 = fopen(fullfile(path, tomIntFilename), 'w'); % Open
fwrite(fId1, tomDownImag, 'single'); % Write
fclose(fId1); % Close
%%
tomDown = abs(tomDownReal+1i*tomDownImag).^2;
tomIntFilename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..512)_y=(1..960)_subsampled.bin';
fId1 = fopen(fullfile(path, tomIntFilename), 'w'); % Open
fwrite(fId1, tomDown, 'single'); % Write
fclose(fId1); % Close
%%
tompathcGAN = 'C:\Users\USER\Documents\GitHub\Fovea\cGAN_1_model_125';
tomDataOverpol0 = load(fullfile(tompathcGAN,"tomDataOverpol0.mat"));
tomDataOverpol0 = tomDataOverpol0.tomDataOver;
tomDataOverpol1 = load(fullfile(tompathcGAN,"tomDataOverpol1.mat"));
tomDataOverpol1 = tomDataOverpol1.tomDataOver;
%%
tomDataOver= zeros(586,896,960,2);
tomDataOver(:,:,:,1) = tomDataOverpol0;
tomDataOver(:,:,:,2) = tomDataOverpol1;
%%
tomDataOverint = abs(tomDataOver).^2;
%%
tomDataOver=sum(tomDataOver,4);

%%
tomIntFilename='[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..960)_y=(1..960)_reconstructed_Real.bin';
fId1 = fopen(fullfile(path, tomIntFilename), 'w'); % Open
fwrite(fId1, real(tomDataOver), 'single'); % Write
fclose(fId1); % Close

tomIntFilename='[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..960)_y=(1..960)_reconstructed_Imag.bin';
fId1 = fopen(fullfile(path, tomIntFilename), 'w'); % Open
fwrite(fId1, imag(tomDataOver), 'single'); % Write
fclose(fId1); % Close
%%
tomIntFilename='[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..960)_y=(1..960)_reconstructed.bin';
fId1 = fopen(fullfile(path, tomIntFilename), 'w'); % Open
fwrite(fId1, tomDataOverint, 'single'); % Write
fclose(fId1); % Close

 