function [ numdel numok ] = idbNumDel(idb,imgIdx)
%
% Count the number of deleted object in an image
% 
% Input: 
%   idb - the image database
%   imgIdx - index of the image in the idb
%
% Output:
%    numdel - number of delete objects
%    numok - number of not deleted objects
%
% This software is provided as is without warranty of any kind. 
% Please report bugs and suggestions to
% uni-heidelberg.keller@daimler.com.


img = idb.images(imgIdx);
objList = img.objList;
numdel = 0;
numok =0;
for o=1:length(objList)
    
    curObjIdx = objList(o);
    
    if idbObjDeleted(idb.objects(curObjIdx)) == true
       numdel = numdel +1;
    else
       numok = numok + 1;
    end
end
