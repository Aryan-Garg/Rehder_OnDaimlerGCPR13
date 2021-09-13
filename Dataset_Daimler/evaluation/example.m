% Create random ground truth and position estimation data
samples = 50;
dims = 2;
gtdata = cell(2,1);
gtdata{1} = randn(dims,samples);
gtdata{2} = randn(dims,samples);
estdata = cell(2,1);
estdata{1} = randn(dims,samples);
estdata{2} = randn(dims,samples);

% Run mean time step wise position calculation
meanposerr = calculateMeanPosErr(cgtdata, cestdata);
fprintf('Mean time step wise position error:\n');
fprintf(' %1.2f\n', meanposerr);

% ---------------------------

% Create random position and prediction errors
P = 32; % prediction steps
N = 122; % number of concatenated positions
all_pos_errs = randn(1,N);
all_pred_errs = cell(P,1);
for i=1:length(all_pred_errs)
    all_pred_errs{i} = randn(1,N);
end

mse = computeWeightedMse(all_pos_errs, all_pred_errs);
fprintf('Weighted MSE: %1.2f\n', mse);

