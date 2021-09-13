function writeContour(fp,obj)
%
% Write the contour information of an object to the file
%   
%
%
% This software is provided as is without warranty of any kind. 
% Please report bugs and suggestions to
% philipp.koelle@daimler.com.

try 
    obj.contour_x;    
catch %#ok<CTCH>
    fprintf(fp, '%s\n', '0');
    return;
end

if isempty(obj.contour_x)
    fprintf(fp, '%s\n', '0');
    return;
end

if length(obj.contour_x) ~= length(obj.contour_y)
    return;
end

fprintf(fp, '%s\n', num2str(length(obj.contour_x)));
for i=1:length(obj.contour_x)
    fprintf(fp, '%s %s ', num2str(obj.contour_x(i)),num2str(obj.contour_y(i)));
    if mod(i, 10) == 0
    fprintf(fp, '\n');       
    elseif i==length(obj.contour_x)
        fprintf(fp, '\n');   
    end
end
%fprintf(fp, '\n');   

    


end