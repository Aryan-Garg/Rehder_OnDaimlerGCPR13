function obj = read2DObject(fid)
% function obj = read2DObject(fid)
%
% Read in a 2D object entry from file identifier.
%
% This software is provided as is without warranty of any kind. 
% Please report bugs and suggestions to
% uni-heidelberg.enzweiler@daimler.com.

obj.data = sparse(17,1);
obj.data(1) = uint8(fscanf(fid,'%d\n',1));
obj.data(2) = uint32(fscanf(fid,'%d',1));
tmp =  uint32(fscanf(fid,'%d\n',1));
if (~isempty(tmp))
    obj.data(3) = tmp;
else
    obj.data(3) = 0;
end
obj.data(4) = single(str2double(fgetl(fid)));

if ( isnan(obj.data(4) ))
    obj.data(4) = -1;
end
%obj.data(5:10) = 0; % only relevant for 3D
obj.data(11)  = int16(fscanf(fid,'%d',1));
obj.data(12)  = int16(fscanf(fid,'%d',1));
obj.data(13)  = int16(fscanf(fid,'%d',1));
obj.data(14)  = int16(fscanf(fid,'%d\n',1));


%% get the contour 
tmp = fgetl(fid);
numcontur = str2num(tmp); %#ok<ST2NM>
contour_x =[];
contour_y =[];
if (~isempty(numcontur) && numcontur >0)
    [ contour_x contour_y ] = readContour(fid, numcontur);
end

%% set additional data
%obj.data(15) = 0;
obj.data(16) = 1;
obj.confVector = obj.data(4);
obj.attributes={};
obj.contour_x = contour_x;
obj.contour_y = contour_y;

% obj.object_class = uint8(fscanf(fid,'%d\n',1));
% obj.obj_id = uint32(fscanf(fid,'%d',1));
% obj.unique_id = uint32(fscanf(fid,'%d\n',1));
% obj.confidence = single(str2double(fgetl(fid)));
% obj.position = sparse(10,1);
% obj.position(1:6,1) = 0; % only relevant for 3D
% obj.position(7,1)  = uint16(fscanf(fid,'%d',1));
% obj.position(8,1)  = uint16(fscanf(fid,'%d',1));
% obj.position(9,1)  = uint16(fscanf(fid,'%d',1));
% obj.position(10,1) = uint16(fscanf(fid,'%d\n',1));
% % eat up one additional line (ignored here)
% ignore        = fscanf(fid,'%g\n',1);
% obj.imgIndex = imgCount;
% obj.has3DPos = 0;
% obj.has2DBox = 1;
% 
% 
% 
% 
