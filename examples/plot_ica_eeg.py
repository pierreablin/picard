"""
=======================================================
Blind source separation using preconditioned ICA on EEG
=======================================================

The examples runs the Picard algorithm proposed in:

Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
Faster independent component analysis by preconditioning with Hessian approximations
ArXiv Preprint, June 2017

"""  # noqa

# Author: Pierre Ablin <pierre.ablin@inria.fr>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from scipy.stats import kurtosis

from picard import picard

print(__doc__)

###############################################################################
# Generate sample EEG data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 40, n_jobs=1)  # 1Hz high pass is often helpful for fitting ICA

picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                       stim=False, exclude='bads')

random_state = 0
data = raw[picks, :][0]
data = data[:, ::2]  # decimate a bit

# Run ICA on data, after reducing the dimension

K, W, Y = picard(data, n_components=30)

###############################################################################
# Plot results
###############################################################################

n_plots = 10
order = np.argsort(kurtosis(Y[:, :1000], axis=1))[::-1]
models = [data[:n_plots], Y[order[:n_plots][::-1]]]
names = ['Observations (raw EEG)',
         'ICA recovered sources']
fig, axes = plt.subplots(2, 1, figsize=(7, 7))
for ii, (model, name, ax) in enumerate(zip(models, names, axes)):
    ax.set_title(name)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    offsets = np.max(model, axis=1) - np.min(model, axis=1)
    offsets = np.cumsum(offsets)
    ax.plot((model[:, :1000] + offsets[:, np.newaxis]).T, 'k')

fig.tight_layout()
plt.show()
