function result = idbObjDeleted(obj)
%
% Check if an image database object has been tagged as deleted
%
% Input: 
%   obj: image database object
%
% Output: 
%   result: true if tagged as deleted
%
% This software is provided as is without warranty of any kind. 
% Please report bugs and suggestions to
% uni-heidelberg.keller@daimler.com.

result = false;
if ( isfield(obj,'del'))
   if (obj.del == true) 
        result = true;
   end
end 