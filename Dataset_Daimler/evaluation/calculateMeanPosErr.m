function [ mean_errs ] = calculateMeanPosErr( gt_data, est_data)
%CALCULATEMEANPOSER This function calculates a time step wise mean position
%error
%   mean_errs = calculateMeanPosErr( gt_data, est_data );
%
% Input:
%   gt_data: Sx1 cell array hold ground truth data for S trajectories. Each
%            cell has a 2xN matrix with lateral and longitudinal positions
%
% Output:
%   mean_errs: 1xN vector holding the time step wise mean position errors
%
% This software is provided as is without warranty of any kind. 
% Please report bugs and suggestions to
% nicolas.schneider@daimler.com.

    % calculate euclidian distance between ground truth and estimated
    % position
    poserrsc = cellfun(@getEuclidDist, gt_data, est_data, 'UniformOutput', false);
    
    % convert to numerical matrix
    poserrs = cell2num(poserrsc);
    
    % calulcate time step wise mean position error
    mean_errs = nanmean(poserrs,1);

end

function [ euclid_dists ] = getEuclidDist( poss1, poss2 )
%GETEUCLIDDIST Computes the euclidian distance between two vectors with
%positions   
%   euclid_dists= getEuclidDist( gt_poss, pred_poss );
%
% Input:
%   poss1: DxN matrix holding N D-dimensional positions
%   poss2: DxN matrix holding N D-dimensional positions
%
% Output:
%   euclid_dists: 1xN vector holding the euclidian distances between each 
%                 position in poss1 and poss2

    euclid_dists = sqrt(sum((poss1 - poss2).^2));

end
