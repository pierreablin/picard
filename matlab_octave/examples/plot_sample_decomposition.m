% This example shows:
% a.) how 'whiten' improves the quality of the decomposition
% b.) the current 'whiten' implementation does not work on rank deficient data

clear variables; clc
%% Define parameters
% PCA parameters
whiten = 1;         % Set to 1 and picard whitens the data

% Test signal
rank_deficient = 1; % Set to 1 and you get a rank deficient matrix

%% Create test signal
signals = {};

% Define independent components
signals{end+1} = @(t) sin(2*pi*t);
signals{end+1} = @(t) 0.1 * randn(size(t));
signals{end+1} = @(t) sawtooth(t, 0.012);

% Create EEG data structure
EEG = eeg_emptyset();
EEG.setname = 'test';
EEG.nbchan = length(signals);
EEG.trials = 1;
EEG.pnts = 2000;
EEG.times = zeros(1, EEG.pnts);
EEG.etc.components = zeros(EEG.nbchan, EEG.pnts);

% Generate the signal
periods = 5;
t = linspace(1, 2*pi*5/7, 2000);

mix_mat = [[1.0 0.2 0.1];
            [0.2 1.0 0.1];
            [0.3 0.2 1.0]];

for var=1:EEG.nbchan,
    EEG.etc.components(var, :) = signals{var}(t);
end

if rank_deficient,
    EEG.nbchan = EEG.nbchan + 1;
    mix_mat = [mix_mat; mix_mat(end,:)];
end

EEG.data = mix_mat * EEG.etc.components;

%% Perform ICA
eeg_rank = rank(EEG.data(1:EEG.nbchan, :));
disp(['data rank is: ' num2str(eeg_rank)])


EEG_ica = pop_runica(EEG, 'icatype', 'picard', 'whiten', whiten);
EEG_ica = eeg_checkset(EEG_ica);
EEG_ica.data = EEG_ica.icawinv * EEG_ica.icaact;

%% Compare reconstruction to original signal
res_string = {'is not', 'is'};
res = all(all(EEG_ica.data - EEG.data < 1e-6));

disp(['Reconstruction ' res_string{res+1} ' the same as the original signal'])

figure
subplot(2, 2, 1)
plot(t, EEG.etc.components)
title('original components')

subplot(2, 2, 2)
plot(t, EEG.data)
title('original signal')

subplot(2, 2, 3)
plot(t, EEG_ica.icaact)
title('found components')

subplot(2, 2, 4)
plot(t, EEG_ica.data)
title('reconstructed signal')