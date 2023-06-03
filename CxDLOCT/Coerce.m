function [matrixCoerced, varargout] = Coerce(matrix, varargin)
  % [matrix, changed] = Coerce(matrix, minimum, maximum)
  % Coerce matrix into range defined by minimum and maximum. By default
  % minimum = 0, maximum = 1.
  % changed is true if any value had to be coerced

  % This Functions follow the coding style that can be
  % sumarized in:
  % * Variables have lower camel case
  % * Functions upper camel case
  % * Constants all upper case
  % * Spaces around operators
  %
  % Authors:  Néstor Uribe-Patarroyo
  %
  % NUP: 
  % 1. Wellman Center for Photomedicine, Harvard Medical School, Massachusetts
  % General Hospital, 40 Blossom Street, Boston, MA, USA;
  % <uribepatarroyo.nestor@mgh.harvard.edu>

  % MGH Flow Measurement project
  %
  % Changelog:
  %
  % V2.0 (2016-11-10): Added handling of nans and optional changed output
  % V1.0 (2015-09-04): Initial version

  minimum = 0;
  maximum = 1;
  if nargin >= 2
    minimum = varargin{1};
  end
  if nargin >= 3
    maximum = varargin{2};
  end
  matrixNanIdx = isnan(matrix);
  matrixCoerced = max(matrix, minimum);
  matrixCoerced = min(matrixCoerced, maximum);
  
  % Put nans back in
  matrixCoerced(matrixNanIdx) = nan;
  
  if nargout > 1
    changed = any(matrix(:) ~= matrixCoerced(:));
    varargout{1} = changed;
  end
end
