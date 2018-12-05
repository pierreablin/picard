%
% =================================================
% Blind source separation using Picard in EEGlab
% =================================================
%

% Author: Pierre Ablin <pierre.ablin@inria.fr>
%         Alexandre Gramfort <alexandre.gramfort@inria.fr>
%         Lukas Stranger <l.stranger@student.tugraz.at>
% License: BSD 3 clause
%
% This example shows how to use picard with the EEGLAB data structure

clear variables; clc
addpath('..//../matlab_octave')
%% Define parameters
% signal
rank_deficient = 1;

% PCA parameters
perform_pca = 1;    % Set to 1 and picard whitens the data with PCA
whiten = 1;

%% Generate sample data
rand('seed', 0);
n_samples = 2000;
time = linspace(0, 8, n_samples);

% Create EEG data structure
EEG = eeg_emptyset();
EEG.setname = 'test';
EEG.nbchan = 3;
EEG.trials = 1;
EEG.pnts = 2000;
EEG.times = zeros(1, EEG.pnts);

% Define independent components
EEG.etc.components(1, :) = sin(2*pi*time);
EEG.etc.components(2, :) = 0.1 * randn(size(time));
EEG.etc.components(3, :) = time;

A = [[1, 1, 1]; [0.5, 2, 1.0]; [1.5, 1.0, 2.0]];  % Mixing matrix
        
if rank_deficient,
    A = [A; A(end,:)];
end

EEG.data = A * EEG.etc.components;

n_comps = rank(A);

%% Perform ICA
if perform_pca,
    EEG_ica = pop_runica(EEG, 'icatype', 'picard', 'whiten', whiten, ...
                         'pca', n_comps);
else
    EEG_ica = pop_runica(EEG, 'icatype', 'picard', 'whiten', whiten);
end

%% Project activations back to sensor space
EEG_ica = eeg_checkset(EEG_ica);
EEG_ica.data = EEG_ica.icawinv * EEG_ica.icaact;

%% Plot results
models = {EEG.data, EEG.etc.components, EEG_ica.icaact, EEG_ica.data};
names = {'Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals with Picard',
         'Reconstructed observations'};

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

res_string = {'is not', 'is'};
res = all(all(EEG_ica.data - EEG.data < 1e-6));

disp(['Reconstruction ' res_string{res+1} ' the same as the original signal'])
