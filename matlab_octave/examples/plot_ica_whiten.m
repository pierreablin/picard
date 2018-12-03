%
% =================================================
% Blind source separation using Picard and Picard-O
% =================================================
%

% Author: Pierre Ablin <pierre.ablin@inria.fr>
%         Alexandre Gramfort <alexandre.gramfort@inria.fr>
% License: BSD 3 clause

% This example shows:
% a.) how 'whiten' improves the quality of the decomposition
%       - set rank_deficient=0 and toggle 'whiten'
% b.) how rank deficiency influences the decomposition
%       - set rank_deficient=1

%% Define parameters
% Test signal
rank_deficient = 0; % Set to 1 and you get a rank deficient matrix

% PCA parameters
whiten = 0;         % Set to 1 and picard whitens the data

%% Generate sample data
rand('seed', 0);
n_samples = 2000;
time = linspace(0, 8, n_samples);

s1 = sin(2*pi*t);
s2 = 0.1 * randn(size(t));
s3 = sawtooth(t, 0.012);

S = [s1; s2; s3];

S = S ./ repmat(std(S, 1, 2), 1, n_samples);  % Standardize data
% Mix data
A = [[1, 1, 1]; [0.5, 2, 1.0]; [1.5, 1.0, 2.0]];  % Mixing matrix

if rank_deficient,
    A = [A; A(end,:)];
end

X = A * S;  % Generate observations

n_sources = size(A, 1);

% Compute ICA
[Y, W] = picard(X, 'whiten', whiten);

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