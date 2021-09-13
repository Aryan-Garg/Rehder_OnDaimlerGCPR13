function [ mse ] = computeWeightedMse( all_pos_errs, all_pred_errs, weights )
%COMPUTEWEIGHTEDMSE Compute weighted MSE of position estimation and prediction
%   mse = computeWeightedMse(all_pos_errs, all_pred_errs, pred_steps, weights);
%   
%   The computeWeightedMse function computes a mean squared error based on
%   the given position estimation errors, position predictions errors and 
%   weights. If weights is empty, uniform weights will be used. The 
%   function can handle NaN values.
%
% Input:
%   all_pos_errs: Nx1 vector holding (euclidian) distances between ground 
%                 truth positions and position estimations (t=0)
%                 concatenated for all trajectories
%   all_pred_errs: Px1 cell array with prediction errors (t=1,...,P), each
%                  cell holding an Nx1 vector with concatenated distances
%                  between ground truth and position prediction
%   pred_steps: number of predictions made into the future (P)
%   weights: (P+1)x1 vector with weights
%
% This software is provided as is without warranty of any kind. 
% Please report bugs and suggestions to
% nicolas.schneider@daimler.com.

    predCnt = numel(all_pred_errs)+1;

    if(nargin < 4)
        weights = zeros(predCnt,1);
        weights = weights + 1/length(weights);
    end
    
    if(predCnt > 1)
        % calculate errors and weight them
        mses = NaN(predCnt,1);
        for p=1:length(all_pred_errs)
            nancnt = sum(isnan(all_pred_errs{p}));
            validcnt = length(all_pred_errs{p})-nancnt;
            mses(p+1) = nansum(all_pred_errs{p}.^2)/validcnt;
        end
        nancnt = sum(isnan(all_pos_errs));
        validcnt = length(all_pos_errs)-nancnt;
        mses(1) = nansum(all_pos_errs.^2)/validcnt;
        mse = sum(mses.*weights);
    else
        nancnt = sum(isnan(all_pos_errs));
        validcnt = length(all_pos_errs)-nancnt;
        mse = nansum(all_pos_errs.^2)/validcnt;
    end

end

