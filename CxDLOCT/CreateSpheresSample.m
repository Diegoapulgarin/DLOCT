function [objPosZ, objPosX, objPosY, amp] = CreateSpheresSample(nZ, nX,...
  nY, SEED, spheresDensity, varType)

rng(SEED); % set defined seed for repeatability
nSpheres = round(spheresDensity * nZ * nX * nY); % number of spheres in volume

% Define radii for each sphere
minRadius = round(min([nZ, nX, nY]) / 32);
maxRadius = round(max([nZ, nX, nY]) / 8);
radii = rand(nSpheres, varType{:}) .* (maxRadius - minRadius) + minRadius;

% Define positions for each sphere (z, x, y)
posSpheres = round(rand(nSpheres, 3, varType{:}) .* [nZ, nX, nY] - [0, nX/2, nY/2] );

% Meshgrid with positions of the spheres
[Z,X,Y] = meshgrid(1:nZ, -nX / 2 : nX / 2 - 1, ...
  -nY / 2 : nY / 2 - 1);
amp = zeros(nZ, nX, nY);

for sphere=1:nSpheres
  spherePoints = (Z - posSpheres(sphere, 1)).^2 + ...
    (X - posSpheres(sphere, 2)).^2 + ...
    (Y - posSpheres(sphere, 3)).^2 < radii(sphere).^2;
  % The spheres reflection coefficient is constant and given by a 
  % random number in [0,1]
  amp(spherePoints) = amp(spherePoints) + rand(1, varType{:});
end

% amp is normalized, as some spheres may overlap
amp = amp ./ max(amp,[],'all');


% positions are flattened, to be given as the algorithm requires
objPosX(1,1,:) = squeeze(X(:));
objPosY(1,1,:) = squeeze(Y(:));
objPosZ(1,1,:) = squeeze(Z(:));

% figure(1), imagesc(amp(:,:,round(nY/2))), axis image
% xlabel('x'), ylabel('z'), title('Sample'), colorbar
% cMap = gray(256);
% colormap(cMap), drawnow

