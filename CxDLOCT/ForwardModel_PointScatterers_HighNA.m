function fringes = ForwardModel_PointScatterers_HighNA(amp, z, x, kVect, xi,...
  alpha, zFP, zRef, maxBatchSize)

  % Number of points
  nPoints = size(z, 3);
  % If not input batchSize calculate contribution from all points at once
  if nargin < 9
    maxBatchSize = nPoints;
  end
  % Batch size
  batchSize = single(min(maxBatchSize, nPoints));
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
    thisBatch = unique(thisBatch,'first');
    % In this case we use sqrt((2*k)^2 - xi^2) where k is a vector
    thisfringes = (1 / (8 * pi) ./ ...
      ((alpha ./ kVect) .^ 2 + (1i * (z(:, :, thisBatch) - zRef) ./ kVect)) .* ...
      sum(exp(-1i * xi .* x(:, :, thisBatch) + ...
      (1i * (z(:, :, thisBatch) - zRef) .* sqrt((2 * kVect) .^ 2 - xi .^ 2)) + ...
      (- (xi * alpha ./ kVect / 2) .^ 2)), 2));
    % sum the contribution of thi batch of scatteres, considering its
    % individual amplitudes, and sum to the overall contribution
    fringes = fringes + sum(amp(:, :, thisBatch) .* thisfringes, 3);
  end
end