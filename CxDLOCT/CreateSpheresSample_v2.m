function objAmp = CreateSpheresSample_v2(sampling, objRange, objPos, ...
  objAmp, layerBounds, spheresRadii, spheresAmps, spheresNumber, varType)

spheresRadii = flip(spheresRadii);
spheresAmps = flip(spheresAmps);
spheresNumber = flip(spheresNumber);

axSampling = sampling(1);
latSampling = sampling(2:3);
% rng(SEED); % set defined seed for repeatability
[nX, nY, nLayers] = size(layerBounds);
nLayers = nLayers - 1; 
objPosZ = objPos(1, :, :);
objPosX = objPos(2, :, :);
objPosY = objPos(3, :, :);
for layer=1:nLayers
  radii = spheresRadii(layer);
  amp = spheresAmps(layer);
  % per layer, calculate the x and y position of the spheres
  posSpheres = round(rand(spheresNumber(layer), 2) .* ...
    [nX - 1, nY - 1]) + 1;
  for sphere=1:spheresNumber(layer)
    % per sphere, calculate z range to belong to layer
    xPos = posSpheres(sphere, 1);
    yPos = posSpheres(sphere, 2);
    zRange = layerBounds(xPos, yPos, layer) - ...
      layerBounds(xPos, yPos, layer + 1) - 2 * radii/axSampling;
    % calculate zPos of sphere
    zPos = zRange * rand(1, varType{:}) + ...
      layerBounds(xPos, yPos, layer + 1) + radii/axSampling;
    % convert center position to sample units
    xPos = (xPos - objRange(2)/2) .* latSampling(1);
    yPos = (yPos - objRange(3)/2) .* latSampling(2);
    zPos = zPos .* axSampling;
    % do binary map to get scatterers inside sphere
    sphereScats = (objPosX - xPos).^2 + ...
    (objPosY - yPos).^2 + ...
    (objPosZ - zPos).^2 < radii.^2;
    objAmp(sphereScats) = objAmp(sphereScats) + amp;
  end
end


