function [fringes] = ForwardModel_PointScatterers_FreqLowNA_3D(amp, z, x, y, kVect, k, xi_x,...
  xi_y, alpha, zFP, zRef, maxBatchSize)
  
  % Beam waist diameter
  beamWaistDiam = 2 * alpha / k;
  % Raylight range
  zR = 2 * alpha ^ 2 / k;
  
  % Remove all points beyond n times the beam position; their contribution
  % is not worth the calculation
  nullPoints = sqrt(x .^ 2 + y .^ 2) > 20 * beamWaistDiam * sqrt(1 + (z / zR) .^ 2);
  amp(nullPoints) = [];
  z(nullPoints) = [];
  x(nullPoints) = [];
  y(nullPoints) = [];

%   objSuscep = zeros(64, 128, 128);
%   [objSuscepIndx, objAmpIndx] = unique(sub2ind(size(objSuscep),...
%     Coerce(round(z / 7.1354e-06), 1, 64),...
%     Coerce(round((x + thisBeamPosX) / (0.1307 * 1e-4)) + 128 / 2, 1, 128),...
%     Coerce(round((y - thisBeamPosY) / (0.1307 * 1e-4)) + 128 / 2, 1, 128)));
%   objSuscep(unique(objSuscepIndx)) = amp(objAmpIndx);
%   figure(1000), imagesc(objSuscep(:, :, 66)), axis image
%   xlabel('x'), ylabel('z'), title('Sample'), colorbar, hold on
  
  % Number of points
  nPoints = size(z, 3);
  % If not input batchSize calculate contribution from all points at once
  if nargin < 10
    maxBatchSize = nPoints;
  end
  % Batch size
  batchSize = min(maxBatchSize, nPoints);
  if isa(kVect,'gpuArray')
    batchSize = gpuArray(batchSize);
  end
  % Number of batches of points
  nBatches = ceil(nPoints / batchSize);
  % Initialize output
  fringes = zeros(size(kVect, 1), 1, 'like', kVect);
  % Iterate batches
  for j = 1:nBatches
    % Calculate the contribution from this batch of points
    thisBatch = min((1:batchSize) + (j - 1) * batchSize, nPoints);
    % In this case we use 2*kVect - xi_x^2/(4*k) where kVect is a vector BUT
    % k is an scalar, yielding the low-NA model
    thisFringes = 1 / (8 * pi .^ 2) ./ ...
      ((alpha ./ k) .^ 2 + (1i * (z(:, :, thisBatch) - zFP) ./ k)) .* ...
      exp(2i * (z(:, :, thisBatch) - zRef) .* kVect) .* ...
      sum(exp(-1i .* ( xi_x .* x(:, :, thisBatch) )) .* ...
      exp(-1i * (z(:, :, thisBatch) - zFP) .* xi_x .^ 2 / k / 4) .* ...
      exp(- (xi_x * alpha / k / 2) .^ 2), 2) .* ...
      sum(exp(-1i .* ( xi_y .* y(:, :, thisBatch) )) .* ...
      exp(-1i * (z(:, :, thisBatch) - zFP) .* xi_y .^ 2 / k / 4) .* ...
      exp(- (xi_y * alpha / k / 2) .^ 2), 4);
    % sum the contribution of all scatteres, considering its individual
    % amplitudes
    fringes = fringes + sum(amp(:, :, thisBatch) .* thisFringes, 3);
  end
end