% Read and display Database format (.db) files.

clear all;
close all;

% read ground truth database
gt_fname = '../Data/TrainingData/2012-04-02_115542/LabelData/gt.db';
disp(['reading ' gt_fname ' ...']);
gt = readImageDatabase(gt_fname);
printImageDatabase(gt);
clear gt;
