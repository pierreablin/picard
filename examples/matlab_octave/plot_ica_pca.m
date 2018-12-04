%
% =================================================
% Blind source separation using Picard and Picard-O
% =================================================
%

% Author: Pierre Ablin <pierre.ablin@inria.fr>
%         Alexandre Gramfort <alexandre.gramfort@inria.fr>
% License: BSD 3 clause

% This example shows:
% a.) how 'pca' is used and its influence onto the decomposition
%       (just toggle perform_pca)

clear variables; clc
addpath('..//../matlab_octave')
%% Define parameters
% PCA parameters
perform_pca = 1;    % Set to 1 and picard whitens the data with PCA

% Fixed parameters 
%   (to show the difference of sphering and pca)
whiten = 1;
rank_deficient = 1;

%% Generate sample data
rand('seed', 0);
n_samples = 2000;
time = linspace(0, 8, n_samples);

s1 = sin(2 * time) .* sin(40 * time);
s2 = sin(3 * time).^5;
s3 = laplace_rnd(size(s1));

S = [s1; s2; s3];

S = S ./ repmat(std(S, 1, 2), 1, n_samples);  % Standardize data
% Mix data
A = [[1, 1, 1]; [0.5, 2, 1.0]; [1.5, 1.0, 2.0]];  % Mixing matrix

if rank_deficient,
    A = [A; A(end,:)];
end

X = A * S;  % Generate observations

n_comps = rank(A);

% Compute ICA
if perform_pca,
    [Y, W] = picard(X, 'whiten', whiten, 'pca', n_comps);
else
    [Y, W] = picard(X, 'whiten', whiten);
end

%% Plot results
models = {X, S, Y};
names = {'Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals with Picard'};

for ii=1:length(models)
    model = models{ii};
    name = names{ii};
    nr_subplots = size(model, 1);
    figure;
    for k=1:nr_subplots
        sig = model(k, :);
        subplot(nr_subplots, 1, k)
        plot(sig);
        if k == 1; title(name); end
    end
end