"""
=======================================================
Blind source separation using preconditioned ICA on EEG
=======================================================

"""
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from sklearn.decomposition import PCA

from lbfgsica import lbfgs_ica

print(__doc__)

###############################################################################
# Generate sample EEG data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 40, n_jobs=1)  # 1Hz high pass is often helpful for fitting ICA

picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                       stim=False, exclude='bads')

random_state = 0
T = 10000
data = raw[picks, :T][0]

# Center
data -= np.mean(data, axis=1, keepdims=True)

# Apply PCA for dimension reduction and whitenning.

N = 30
pca = PCA(n_components=N, whiten=True, svd_solver='full')
pca.fit(data)
X = pca.components_ * np.sqrt(float(T))

# Run ICA on X

Y, W = lbfgs_ica(X, maxiter=1000, verbose=False)

###############################################################################
# Plot results
N_plots = 10
models = [X[:N_plots], Y[:N_plots]]
names = ['Observations (raw EEG)',
         'ICA recovered sources']
colors = ['k', 'k']
fig, axes = plt.subplots(2, 1, figsize=(6, 4))
for ii, (model, name, ax) in enumerate(zip(models, names, axes)):
    ax.set_title(name)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    offset = 0.
    for i in range(N_plots):
        sig = model[i]
        ax.plot(sig + offset, color=colors[ii])
        offset += np.max(sig) - np.min(sig)
plt.show()
