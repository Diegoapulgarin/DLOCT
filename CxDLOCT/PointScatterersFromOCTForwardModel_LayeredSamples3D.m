% This is a sample simulation to generate a B-scan image from an synthetic
% object using OCT imaging model

% Paths to use
addpath(genpath('..\..\..\matlab'));

% Some options for figures
LATEX_DEF = {'Interpreter', 'latex'};
set(0, 'defaultTextInterpreter', 'LaTex')
set(0, 'defaultAxesTickLabelInterpreter', 'LaTex')
set(groot,'defaultLegendInterpreter','latex');
set(0, 'DefaultAxesFontSize', 20)

% Number of figure to open
curFig = 10;

%% Parameters of the simulation
% Tomogram number of points
nZ = 512; % axial, number of pixels per A-line, accounting for zero-padding
nX = 256 + 32; % fast scan axis, number of A-lines per B-scan
nY = 1; % slow scan axis, number of B-scans per tomogram
nK = 400; % Number of samples, <= nZ, difference is zero padding
xNyquistOversampling = 1; % Galvo sampling factor. 1->Nyquist
nXOversampling = nX; % Number of Alines for oversampling PSF <= nX, difference is zero padding

useGPU = false;

% Spectral parameters
wavelength = 1.310e-6; % Source central wavelength
wavelengthWidthSource = 1*120e-9; % Spectral full width at half maximum
axialRes = 2 * log(2) / pi * wavelength ^ 2 / wavelengthWidthSource; % Axial resolution
% wavelengthWidthSource = 2 * log(2) / pi * wavelength ^ 2 / axialRes;

% Confocal parameters
numAper = 0.05; % clcNumerical aperture

% Noise floor level in dB
noiseFloorDb = 20;

if useGPU
  varType = {'single', 'gpuArray'};
  ThisLinspace = @gpuArray.linspace;
  ToSingle = @(x) gpuArray(single(x));
else
  varType = {'single'};
  ThisLinspace = @linspace;
  ToSingle = @single;
end

% Wavenumber spectral range
wavenumberRange = 2 * pi ./ (wavelength + ([wavelengthWidthSource -wavelengthWidthSource] / 2));
% Wavenumber sampling vector. Because we are simulating complex fringes, we
% need nZ and not 2 * nZ
zeroPadding = (nZ - nK) / 2;
kVect = ThisLinspace(wavenumberRange(1), wavenumberRange(2), nK)';
wavenumber = single((wavenumberRange(1) + wavenumberRange(2)) / 2);

% Wavenumber spectral width of the source
wavenumberWidthSource = 2 * pi / (wavelength - (wavelengthWidthSource / 2)) - ...
  2 * pi / (wavelength + (wavelengthWidthSource / 2)); 
% Linear in wavenumber spectrum of the source
wavenumberFWHMSource = wavenumberWidthSource / (2 * sqrt(2 * log(2)));
sourceSpec = exp(-(kVect - wavenumber) .^ 2 / 2 / wavenumberFWHMSource ^ 2);
% figure(1), plot(kVect, sourceSpec), axis tight

% Physical size of axial axis
zSize = pi * nK / diff(wavenumberRange);
% zSize = wavelength ^ 2 * nK / (4 * wavelengthWidthSource); % Alternatively
axSampling = zSize / nZ; % Axial sampling

% Confocal parameters
alpha = pi / numAper; % Confocal constant
beamWaistDiam = 2 * alpha / wavenumber; % Beam waist diameter 
latSampling = beamWaistDiam / 2 / xNyquistOversampling; % Latereral sampling
confocalParm = pi * (beamWaistDiam / 2) ^ 2 / wavelength; % Confocal paremeter (for info.)

% Zero-path delay. Changing this changes the focal plane in the HighNA
% model
zRef = zSize / 2;
% Distance from the top plane to the focal plane. This is independent from
% zRef only for the LowNA models and not for the HighNA model
focalPlane = zSize / 4;

xSize = latSampling * nX; % Physical size of fast scan axis
ySize = latSampling * nY; % Physical size of slow scan axis

% Coordinates
% Cartesian coordinate
zVect = single(ThisLinspace(0, zSize - axSampling, nZ));
xVect = single(ThisLinspace(-xSize / 2, xSize / 2 - latSampling, nX));
yVect = ThisLinspace(- ySize / 2, ySize / 2 - latSampling, nY); single(0); 

% Frequency coordinates
freqBWFac = 2; % Increase frequency bandwidth to avoid artifact in numerical FT
nFreqX = nX * freqBWFac;
freqXVect = single(ThisLinspace(-0.5, 0.5 - 1 / nFreqX, nFreqX)) /...
  (latSampling / freqBWFac) * 2 * pi;

nFreqY = nY * freqBWFac;
clear freqYVect
freqYVect(1,1,1,:) = single(ThisLinspace(-0.5, 0.5 - 1 / nFreqY, nFreqY)) /...
  (latSampling / freqBWFac) * 2 * pi;


%% Create object with point scatteres
% rng('default')
% Number of point scatterers
nPointSource = nY * 50000; % be sure that nPointsSource/nY is integer
maxPointsBatch = round(nPointSource/16);

% Range where point scatteres appear
layersPointSources = 0; [4 6];
zStart = 32;
zOffset = 0;
objZRange = nZ - 32;
objXRange = nX - 32;
objYRange = nY - 0;
objRange = [objZRange, objXRange];
layeredObj = true;

if layeredObj
  % Porcentaje length of each layer
  layerPrcts = [5 20 5 10 5 5 2 5 43];
  % Signal strength of each layer
  layerBackScat = [5 1 0.5 1 0.5 10 5 10 5] * 1e-4; [1 1 5 2 4 1] * 1e-4; % [1 1 1 100] * 1e-4; 
  layerScat = [1 1 1 1 1 1 1 1 1] / 2; [1 5 2 1];
  % Sampling vector
  sampling = [axSampling, latSampling];
  % Created layered sample
  [objPos, objAmp, layerBounds] = CreateLayeredSample(layerPrcts,...
    layerBackScat, layerScat, layersPointSources, nPointSource/nY, objRange,...
    zStart, sampling, varType, maxPointsBatch);
  objPosZ = objPos(1, :, :) - zOffset * axSampling;
  objPosX = objPos(2, :, :);
%   objPosZ(1, 1, 1:10) = linspace(min(objPosZ), max(objPosZ), 10);
%   objPosX(1, 1, 1:10) = 0;
%   objAmp(1, 1, 1:10) = 10; objAmp(1, 1, [2 10]) = 0;
else
  % Because of complex fringes, z range is defined as [-nZ / 2, nZ / 2] * axSampling
  % objPosZ = permute(linspace(- objZRange / 2, objZRange / 2, nPointSource) * axSampling, [1 3 2]);
  objPosZ = (objZRange * (rand(1, 1, nPointSource, varType{:})) - objZRange / 2) * axSampling;
  objPosX = (objXRange * (rand(1, 1, nPointSource, varType{:})) - objXRange / 2) * latSampling;
  
  % Amplitude of scatterers
  objAmp = ones(1, 1, nPointSource, varType{:}); % All ones by default
end

% Just to visualize point scatterers, offset to nZ / 2 to index matrix
% correctly due to complex fringes. Coerce is important because sometimes
% we will get an object that rounds to 0 index.
objSuscep = zeros(nZ, nX, nY, varType{:});
[objSuscepIndx, objAmpIndx] = unique(sub2ind(size(objSuscep), Coerce(round(objPosZ / axSampling), 1, nZ),...
  Coerce(round(objPosX / latSampling) + nX / 2, 1, nX)));
objSuscep(unique(objSuscepIndx)) = objAmp(objAmpIndx);

% Adding next B-scans as copies of the initial one
clear objPosY
objPosY(1,1,:) = reshape(yVect .* ones(nPointSource / nY, nY), ...
    [1, nPointSource]);
for i=2:nY
  objAmp = cat(3,objAmp, objAmp(:,:,1:nPointSource/nY));
  objPosX = cat(3,objPosX, objPosX(:,:,1:nPointSource/nY));
  objPosZ = cat(3,objPosZ, objPosZ(:,:,1:nPointSource/nY));
  objSuscep(:,:,i) = objSuscep(:,:,1);
end

% Show point scatterers
for i=1:1
  figure(curFig +1), imagesc(objSuscep(:, :, i)), axis image
  xlabel('x'), ylabel('z'), title('Sample'), colorbar, hold on
  layerBoundsFull = padarray(layerBounds(:, 1:end) - zOffset, [(nX - objRange(2)) / 2, 0], nan, 'both');
  plot(layerBoundsFull, 'k', 'LineWidth', 1), hold off
  curFig = curFig + 1;
  cMap = viridis(256);
  colormap(cMap), drawnow
end
curFig = curFig - 1; 

%% Forward Model
modelISAM = false;
tic
% High NA Model
fringes1 = zeros(nK, nX, nY, varType{:});
if modelISAM
  for thisScan = 1:nX
    % Current beam position
    thisBeamPosX = xVect(thisScan);
    % Spectrum at this beam possition is the contribution of the Gaussian
    % beam at the location of the point sources
    fringes1(:, thisScan) =  ForwardModel_PointScatterers_HighNA(objAmp, objPosZ,...
      objPosX - thisBeamPosX, kVect, freqXVect,...
      alpha, focalPlane, zRef);
  end
else
  % Low NA Model
  for thisYScan = 1:nY
    for thisXScan = 1:nX
      % Current beam position
      thisBeamPosX = xVect(thisXScan);
      thisBeamPosY = yVect(thisYScan);
      % Spectrum at this beam possition is the contribution of the Gaussian
      % beam at the location of the point sources
      fringes1(:, thisXScan, thisYScan) =  ForwardModel_PointScatterers_FreqLowNA_3D(objAmp, objPosZ,...
        objPosX - thisBeamPosX, objPosY - thisBeamPosY, kVect, wavenumber, freqXVect,...
        freqYVect, alpha, focalPlane, zRef, maxPointsBatch);
      
    end
    
  end
end
% Calculate fringes with proper constants, including source spectrum
fringes1 = fringes1 .* 1i ./ ((2 * pi) .^ 2) .* 1 .* sqrt(sourceSpec) ./ kVect;
toc

%% Fringes processing
% Fourier transform fringes to get tomogram
% Apply Hanning to avoid artifacts
tom1True = fftshift(fft(fftshift(padarray(fringes1 .* hanning(nK), zeroPadding, 'both'), 1), [], 1), 1);
tom1 = fftshift(fft(fftshift(padarray(fringes1 .* hanning(nK), zeroPadding, 'both'), 1), [], 1), 1) +...
  (((10 ^ (noiseFloorDb / 20)) / 1) * (randn(nZ, nX, nY) + 1i * randn(nZ, nX, nY)));

% Aline are shifted depending on the zero-path delay
refShift = round((2 * zRef + zSize) / zSize * nZ);
tom1 = circshift(tom1, [refShift / 2 0]);
tom1True = circshift(tom1True, [refShift / 2 0]);

% tom1 = circshift(tom1, [128 0]);
% 10 * log10(mean(abs(tom1(1:10, 1:50)) .^ 2, 'all'))

% Show OCT images
% Plot optioncs
logLim = [60 inf];
cMap = (gray(256));

% Sample and tomogram intensity
figH1 = figure(curFig + 2); subplot(1,2,1), imagesc(squeeze(xVect) * 1e6, squeeze(flip(zVect))* 1e6,...
  objSuscep(:, :, 1)), axis image
xlabel('$x$ [$\mu$m]'), ylabel('$z$ [$\mu$m]'), title('(a) Sample'),
set(gca,'YDir','normal'), colormap(cMap), drawnow,
hCB = colorbar; hCB.TickLabelInterpreter = 'latex'; %hCB.Ticks = linspace(0, 4, 6);
hCB.Label.String = '[a.u.]'; hCB.Label.Interpreter = 'latex'; hCB.Label.FontSize = 20;

subplot(1,2,2), imagesc(squeeze(xVect) * 1e6, squeeze(zVect)* 1e6, 10 * log10(abs(tom1(:,:,1)) .^ 2), logLim),
xlabel('$x$ [$\mu$m]'), title('(b) OCT image'), axis image % ylabel('$z$ [$\mu$m]'), 
set(gca, 'YTick', [])
hCB = colorbar; hCB.TickLabelInterpreter = 'latex'; hCB.Label.String = 'log. scale [a.u.]';
hCB.Label.Interpreter = 'latex';  hCB.Label.FontSize = 20;

% Tomogram phase, should be like speckle
figH2 = figure(curFig + 3);
imagesc(squeeze(xVect) * 1e6, squeeze(zVect)* 1e6, angle(tom1), [-pi pi]),
xlabel('$x$ [$\mu$m]'), title('OCT phase image'), axis image % ylabel('$z$ [$\mu$m]'), 
set(gca, 'YTick', []), colormap(cmap('c3')),
hCB = colorbar; hCB.TickLabelInterpreter = 'latex'; hCB.Label.String = 'rad';
hCB.Label.Interpreter = 'latex';  hCB.Label.FontSize = 20;

% Tomogram phase diff, should tend to zero (roughly dark brown)
figH2 = figure(curFig + 4);
imagesc(squeeze(xVect) * 1e6, squeeze(zVect)* 1e6, angle(tom1(:, 2:end) .* conj(tom1(:, 1:end-1))), [-pi pi]),
xlabel('$x$ [$\mu$m]'), title('OCT phase image'), axis image % ylabel('$z$ [$\mu$m]'), 
set(gca, 'YTick', []), colormap(cmap('c3')),
hCB = colorbar; hCB.TickLabelInterpreter = 'latex'; hCB.Label.String = 'rad';
hCB.Label.Interpreter = 'latex';  hCB.Label.FontSize = 20;

% MPS, should be Gaussian with sufficient bandwidth to capture the fall to
% ground value
tom1MPS = CalcTomFTXYMean(tom1, permute(hanning(nX), [3 1 2]), permute(hanning(nY), [3 2 1]), 0, 1);
figure2(curFig + 5), plot(tom1MPS, 'k', 'lineWidth', 2),
ylabel('MPS [a.u]'), xlabel('X Freq [px]'),
