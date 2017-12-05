%
% =================================================
% Blind source separation using Picard and Picard-O
% =================================================
%

% Author: Pierre Ablin <pierre.ablin@inria.fr>
%         Alexandre Gramfort <alexandre.gramfort@inria.fr>
% License: BSD 3 clause

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
X = A * S;  % Generate observations

n_sources = size(A, 1);

% Compute ICA
picard(X);
[Y_picard, W_picard] = picard(X);
% [Y_picardo, W_picardo] = picardo(X);

%% Plot results

models = {X, S, Y_picard};
names = {'Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals with Picard'};
% models = {X, S, Y_picard, Y_picardo};
% names = {'Observations (mixed signal)',
%          'True Sources',
%          'ICA recovered signals with Picard',
%          'ICA recovered signals with Picard-O'};

for ii=1:length(models)
    model = models{ii};
    name = names{ii};
    figure;
    for k=1:n_sources
        sig = model(k, :);
        subplot(3, 1, k)
        plot(sig);
        if k == 1; title(name); end
    end
end
