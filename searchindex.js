Search.setIndex({"docnames": ["api", "auto_examples/index", "auto_examples/plot_custom_density", "auto_examples/plot_ecg_detection", "auto_examples/plot_faces_decomposition", "auto_examples/plot_ica_eeg", "auto_examples/plot_ica_synth", "auto_examples/sg_execution_times", "generated/picard.Picard", "generated/picard.amari_distance", "generated/picard.permute", "generated/picard.picard", "index"], "filenames": ["api.rst", "auto_examples/index.rst", "auto_examples/plot_custom_density.rst", "auto_examples/plot_ecg_detection.rst", "auto_examples/plot_faces_decomposition.rst", "auto_examples/plot_ica_eeg.rst", "auto_examples/plot_ica_synth.rst", "auto_examples/sg_execution_times.rst", "generated/picard.Picard.rst", "generated/picard.amari_distance.rst", "generated/picard.permute.rst", "generated/picard.picard.rst", "index.rst"], "titles": ["API Documentation", "Examples Gallery", "Using a custom density with Picard", "Comparing Picard and FastICA for the task of detecting ECG artifacts in MEG", "Comparison of Picard-O and FastICA on faces data", "Blind source separation using preconditioned ICA on EEG", "Blind source separation using Picard and Picard-O", "Computation times", "picard.Picard", "picard.amari_distance", "picard.permute", "picard.picard", "Picard"], "terms": {"function": [0, 2, 3, 5, 8], "class": [0, 2, 8, 11], "us": [1, 3, 4, 7, 8, 11, 12], "custom": [1, 7, 8, 11], "densiti": [1, 7, 8, 11], "picard": [1, 5, 7], "blind": [1, 7], "sourc": [1, 2, 3, 4, 7, 8, 11, 12], "separ": [1, 7, 8, 11], "o": [1, 5, 7, 8, 11, 12], "precondit": [1, 6, 7, 12], "ica": [1, 3, 4, 6, 7, 12], "eeg": [1, 3, 7, 12], "compar": [1, 4, 7], "fastica": [1, 7, 11, 12], "task": [1, 7], "detect": [1, 7], "ecg": [1, 7], "artifact": [1, 7], "meg": [1, 5, 7], "comparison": [1, 7], "face": [1, 7], "data": [1, 3, 5, 6, 7, 8, 11, 12], "download": [1, 2, 3, 4, 5, 6], "all": [1, 8], "python": [1, 2, 3, 4, 5, 6, 12], "code": [1, 2, 3, 4, 5, 6, 12], "auto_examples_python": 1, "zip": [1, 2, 4, 5, 6], "jupyt": [1, 2, 3, 4, 5, 6], "notebook": [1, 2, 3, 4, 5, 6], "auto_examples_jupyt": 1, "gener": [1, 2, 3, 4, 5, 6, 11], "sphinx": [1, 2, 3, 4, 5, 6], "click": [2, 3, 4, 5, 6], "here": [2, 3, 4, 5, 6], "full": [2, 3, 4, 5, 6], "exampl": [2, 3, 4, 5, 6, 8, 11, 12], "thi": [2, 3, 4, 8, 11, 12], "show": [2, 4, 5, 6, 12], "how": 2, "author": [2, 3, 4, 5, 6], "pierr": [2, 3, 4, 5, 6, 12], "ablin": [2, 3, 4, 5, 6, 12], "inria": [2, 4, 5, 6], "fr": [2, 3, 4, 5, 6], "alexandr": [2, 3, 4, 5, 6, 12], "gramfort": [2, 3, 4, 5, 6, 12], "licens": [2, 3, 4, 5, 6], "bsd": [2, 3, 4, 5, 6], "3": [2, 3, 4, 5, 6, 12], "claus": [2, 3, 4, 5, 6], "import": [2, 3, 4, 5, 6, 8, 12], "numpi": [2, 4, 5, 6, 12], "np": [2, 4, 5, 6, 8, 11, 12], "matplotlib": [2, 4, 5, 6, 12], "pyplot": [2, 4, 5, 6], "plt": [2, 4, 5, 6], "from": [2, 3, 4, 5, 6, 8, 10, 11, 12], "permut": [2, 9], "print": [2, 4, 5, 6, 11], "__doc__": [2, 4, 5, 6], "build": 2, "where": [2, 11], "score": [2, 3], "i": [2, 3, 4, 5, 8, 9, 11, 12], "x": [2, 4, 5, 6, 8, 11, 12], "tanh": [2, 4, 8, 11], "customdens": 2, "object": [2, 5], "def": [2, 3, 4], "log_lik": [2, 8, 11], "self": [2, 8], "y": [2, 4, 5, 11, 12], "return": [2, 4, 5, 9, 10, 11], "2": [2, 4, 5, 6, 8, 11, 12], "log": 2, "cosh": 2, "score_and_d": [2, 8, 11], "tanhi": 2, "custom_dens": 2, "plot": [2, 3, 4, 5, 6], "correspond": [2, 4, 8, 11], "linspac": [2, 6], "100": [2, 5], "log_likelihood": 2, "psi": 2, "psi_der": 2, "name": [2, 4, 5, 6], "likelihood": 2, "deriv": 2, "figur": [2, 4], "valu": 2, "label": [2, 4], "legend": [2, 4], "titl": [2, 3], "run": [2, 3, 4, 5, 6, 11, 12], "toi": 2, "dataset": [2, 3, 4, 5, 8, 12], "rng": [2, 4], "random": [2, 4, 6, 11, 12], "randomst": [2, 4, 8, 11], "0": [2, 3, 4, 5, 6, 7, 8, 11], "n": [2, 11, 12], "t": [2, 4, 5, 6, 11, 12], "5": [2, 3, 4, 5, 6], "1000": [2, 5, 12], "": [2, 3, 5, 6, 8, 11, 12], "laplac": [2, 6, 12], "size": [2, 6, 8, 11, 12], "A": [2, 6, 8, 9, 10, 11, 12], "randn": [2, 12], "dot": [2, 4, 6, 8, 11, 12], "k": [2, 4, 5, 11, 12], "w": [2, 4, 5, 9, 11, 12], "fun": [2, 8, 11], "random_st": [2, 3, 4, 5, 6, 8, 11], "imshow": 2, "interpol": 2, "nearest": 2, "product": 2, "between": [2, 9], "estim": [2, 11, 12], "unmix": [2, 8, 12], "matrix": [2, 6, 8, 9, 10, 11, 12], "mix": [2, 6, 8, 11], "total": [2, 3, 4, 5, 6, 7], "time": [2, 3, 4, 5, 6], "script": [2, 3, 4, 5, 6, 12], "minut": [2, 3, 4, 5, 6], "296": [2, 7], "second": [2, 3, 4, 5, 6], "plot_custom_dens": [2, 7], "py": [2, 3, 4, 5, 6, 7, 8, 11], "ipynb": [2, 3, 4, 5, 6], "galleri": [2, 3, 4, 5, 6], "ar": [3, 8, 12], "fit": [3, 5, 8], "sever": 3, "initi": [3, 8, 11], "The": [3, 5, 6, 8, 9, 11, 12], "relat": 3, "displai": [3, 4], "each": [3, 4, 8], "faster": [3, 4, 5, 6, 12], "less": 3, "depend": 3, "pierreablin": 3, "gmail": 3, "com": [3, 4], "telecom": 3, "paristech": 3, "mne": [3, 5, 12], "preprocess": [3, 11], "create_ecg_epoch": 3, "sampl": [3, 5, 6, 11], "setup": 3, "path": [3, 5], "prepar": 3, "raw": [3, 5], "set_log_level": 3, "verbos": [3, 11], "critic": 3, "data_path": [3, 5], "raw_fnam": [3, 5], "sample_audvis_filt": [3, 5], "40_raw": [3, 5], "fif": [3, 5], "io": [3, 5], "read_raw_fif": [3, 5], "preload": [3, 5], "true": [3, 4, 5, 6, 8, 10, 11, 12], "filter": [3, 5], "1": [3, 4, 5, 6, 12], "none": [3, 4, 8, 11], "fir_design": 3, "firwin": [3, 5], "alreadi": [3, 8, 11], "lowpass": 3, "40": [3, 4, 5, 6, 7], "set_annot": 3, "annot": 3, "10": [3, 4, 5, 8, 11], "bad": [3, 5], "block": 3, "For": 3, "sake": 3, "we": [3, 8, 12], "first": [3, 8, 11, 12], "record": 3, "part": 3, "exclud": [3, 5], "decomposit": [3, 4, 12], "default": [3, 4, 5, 8, 11], "To": [3, 12], "turn": 3, "behavior": [3, 8, 11, 12], "off": 3, "pass": [3, 5, 8], "reject_by_annot": 3, "fals": [3, 5, 6, 8, 11], "meth": 3, "pick": [3, 5], "pick_typ": [3, 5], "info": [3, 5], "eog": [3, 5], "stim": [3, 5], "defin": 3, "identifi": 3, "compon": [3, 6, 8, 11, 12], "result": [3, 4, 5, 6, 8, 11], "topographi": 3, "fit_ica_and_plot_scor": 3, "method": [3, 5, 8, 11], "tol": [3, 4, 8, 11], "1e": [3, 4, 8, 11], "4": [3, 5, 6], "n_compon": [3, 4, 5, 8, 11, 12], "95": [3, 5], "fit_param": 3, "max_it": [3, 4, 8, 11], "400": [3, 4], "t0": [3, 4], "copi": [3, 12], "decim": [3, 5], "reject": 3, "dict": [3, 4], "mag": 3, "4e": 3, "12": [3, 5], "grad": 3, "4000e": 3, "13": [3, 5], "warn": [3, 4], "fit_tim": 3, "init": 3, "d": [3, 4, 9], "took": 3, "2g": 3, "sec": [3, 4, 5], "ecg_epoch": 3, "tmin": 3, "tmax": 3, "ecg_ind": 3, "find_bads_ecg": 3, "ctp": 3, "plot_scor": 3, "plot_compon": 3, "colorbar": 3, "few": 3, "iter": [3, 4, 8, 11], "differ": [3, 4, 8, 11], "n_init": 3, "rang": [3, 5], "home": [3, 4, 5], "circleci": [3, 4, 5], "local": [3, 4], "lib": [3, 4], "python3": [3, 4], "8": [3, 4, 5, 6, 12], "site": [3, 4], "packag": [3, 4], "sklearn": [3, 4, 8, 12], "_fastica": [3, 4], "120": [3, 4], "convergencewarn": [3, 4], "did": [3, 4], "converg": [3, 4, 11, 12], "consid": [3, 4, 8, 11], "increas": [3, 4], "toler": [3, 4, 8, 11], "maximum": [3, 4, 8, 11], "number": [3, 4, 8, 11], "do": [3, 5, 12], "same": 3, "thing": 3, "third": 3, "found": 3, "doe": 3, "realli": 3, "look": 3, "like": [3, 11], "an": [3, 8, 11], "consist": [3, 4], "across": [3, 8], "find": 3, "more": 3, "086": [3, 7], "plot_ecg_detect": [3, 7], "jean": [4, 5, 6, 12], "fran\u00e7oi": [4, 5, 6, 12], "cardoso": [4, 5, 6, 12], "under": [4, 5, 6, 12], "orthogon": [4, 5, 6, 8, 12], "constraint": [4, 5, 6, 8, 12], "icassp": [4, 5, 6, 12], "2018": [4, 5, 6, 12], "http": [4, 5, 6, 12], "arxiv": [4, 5, 6, 12], "org": [4, 5, 6, 12], "ab": [4, 5, 6, 12], "1711": [4, 5, 6, 12], "10873": [4, 5, 6, 12], "On": 4, "abov": 4, "bar": 4, "final": 4, "gradient": 4, "norm": [4, 11], "fetch_olivetti_fac": 4, "image_shap": 4, "64": [4, 5], "gradient_norm": 4, "psii": 4, "psidy_mean": 4, "mean": [4, 8, 11, 12], "axi": [4, 5, 6], "g": 4, "shape": [4, 6, 8, 9, 10, 11, 12], "sign": 4, "diag": 4, "linalg": [4, 11], "load": 4, "shuffl": 4, "n_sampl": [4, 6, 11], "n_featur": [4, 8, 9, 10, 11], "global": 4, "center": [4, 11], "faces_cent": 4, "reshap": 4, "olivetti": 4, "ndownload": 4, "figshar": 4, "file": [4, 5, 7, 8, 11], "5976027": 4, "scikit_learn_data": 4, "store": 4, "elaps": 4, "dimens": [4, 5, 11], "60": [4, 5], "algorithm": [4, 5, 6, 8, 11, 12], "picardo": 4, "color": [4, 6], "b": [4, 5], "orang": [4, 6], "running_tim": 4, "kwarg": 4, "500": [4, 8, 11], "els": 4, "append": 4, "494": 4, "futurewarn": 4, "start": [4, 11, 12], "v1": [4, 5], "whiten": [4, 8, 11, 12], "unit": 4, "varianc": 4, "autolabel": 4, "rect": 4, "gradient_list": 4, "attach": 4, "text": 4, "its": [4, 10, 12], "height": 4, "get_height": 4, "g_string": 4, "6": [4, 5, 6], "ax": [4, 5, 6], "get_x": 4, "get_width": 4, "fontsiz": 4, "ha": 4, "va": 4, "bottom": 4, "fig": [4, 5, 6], "subplot": [4, 5, 6], "ind": 4, "arang": 4, "len": 4, "width": 4, "enumer": [4, 5, 6], "05": [4, 5, 7], "set_xtick": 4, "set_xticklabel": 4, "str": [4, 8, 11], "xlabel": 4, "ylabel": 4, "33": [4, 5, 7], "073": [4, 7], "plot_faces_decomposit": [4, 7], "propos": [5, 6], "scipi": [5, 12], "stat": 5, "kurtosi": 5, "n_job": 5, "1hz": 5, "high": 5, "often": 5, "help": [5, 11], "bit": 5, "locat": 5, "mne_data": 5, "creat": 5, "process": [5, 6, 12], "tar": 5, "gz": 5, "osf": 5, "86qa2": 5, "version": [5, 12], "00": [5, 7], "65g": 5, "56m": 5, "29": 5, "55": 5, "6mb": 5, "7m": 5, "25": 5, "65": 5, "1mb": 5, "20": 5, "0m": 5, "23": [5, 12], "68": 5, "4mb": 5, "27": 5, "4m": 5, "22": 5, "70": 5, "7mb": 5, "34": 5, "71": 5, "41": 5, "9m": 5, "9mb": 5, "49": 5, "1m": 5, "56": 5, "5m": 5, "69": 5, "72": 5, "0mb": 5, "3m": 5, "01": [5, 8, 11], "67": 5, "3mb": 5, "78": 5, "85": 5, "8m": 5, "93": 5, "21": 5, "101m": 5, "5mb": 5, "7": [5, 8, 11, 12], "108m": 5, "73": 5, "116m": 5, "74": 5, "124m": 5, "131m": 5, "139m": 5, "9": 5, "146m": 5, "02": 5, "2mb": 5, "154m": 5, "161m": 5, "169m": 5, "19": [5, 12], "11": 5, "176m": 5, "183m": 5, "190m": 5, "198m": 5, "205m": 5, "62": 5, "211m": 5, "03": [5, 7], "219m": 5, "66": 5, "14": [5, 12], "226m": 5, "28": 5, "50": 5, "233m": 5, "15": 5, "240m": 5, "24": 5, "58": 5, "8mb": 5, "247m": 5, "61": 5, "254m": 5, "16": 5, "261m": 5, "268m": 5, "17": 5, "275m": 5, "04": 5, "282m": 5, "18": 5, "290m": 5, "297m": 5, "305m": 5, "312m": 5, "320m": 5, "327m": 5, "335m": 5, "343m": 5, "75": 5, "350m": 5, "358m": 5, "365m": 5, "373m": 5, "381m": 5, "388m": 5, "396m": 5, "403m": 5, "411m": 5, "419m": 5, "26": 5, "426m": 5, "06": [5, 7], "434m": 5, "442m": 5, "76": 5, "449m": 5, "457m": 5, "464m": 5, "472m": 5, "480m": 5, "487m": 5, "30": 5, "495m": 5, "502m": 5, "07": [5, 8, 11], "31": 5, "510m": 5, "517m": 5, "32": 5, "525m": 5, "532m": 5, "540m": 5, "548m": 5, "555m": 5, "563m": 5, "35": 5, "570m": 5, "08": 5, "578m": 5, "585m": 5, "36": 5, "593m": 5, "601m": 5, "37": 5, "608m": 5, "616m": 5, "38": 5, "623m": 5, "631m": 5, "39": 5, "638m": 5, "646m": 5, "09": 5, "654m": 5, "661m": 5, "669m": 5, "676m": 5, "683m": 5, "42": 5, "691m": 5, "699m": 5, "43": 5, "706m": 5, "714m": 5, "44": 5, "721m": 5, "729m": 5, "45": 5, "736m": 5, "743m": 5, "751m": 5, "46": 5, "759m": 5, "766m": 5, "47": 5, "774m": 5, "782m": 5, "48": 5, "789m": 5, "797m": 5, "805m": 5, "812m": 5, "820m": 5, "828m": 5, "51": 5, "835m": 5, "843m": 5, "851m": 5, "52": [5, 7], "858m": 5, "866m": 5, "53": 5, "874m": 5, "882m": 5, "54": 5, "889m": 5, "897m": 5, "905m": 5, "77": 5, "912m": 5, "920m": 5, "928m": 5, "57": 5, "936m": 5, "943m": 5, "951m": 5, "959m": 5, "966m": 5, "59": 5, "974m": 5, "982m": 5, "989m": 5, "997m": 5, "00g": 5, "01g": 5, "02g": 5, "03g": 5, "63": 5, "04g": 5, "05g": 5, "06g": 5, "07g": 5, "08g": 5, "09g": 5, "10g": 5, "11g": 5, "12g": 5, "13g": 5, "14g": 5, "15g": 5, "16g": 5, "17g": 5, "18g": 5, "19g": 5, "20g": 5, "21g": 5, "22g": 5, "23g": 5, "24g": 5, "25g": 5, "26g": 5, "27g": 5, "28g": 5, "29g": 5, "30g": 5, "79": 5, "31g": 5, "80": 5, "32g": 5, "33g": 5, "81": 5, "34g": 5, "82": 5, "35g": 5, "36g": 5, "83": 5, "37g": 5, "38g": 5, "84": 5, "39g": 5, "40g": 5, "41g": 5, "86": 5, "42g": 5, "87": 5, "43g": 5, "44g": 5, "45g": 5, "88": 5, "46g": 5, "89": 5, "47g": 5, "48g": 5, "90": 5, "49g": 5, "91": 5, "50g": 5, "51g": 5, "92": 5, "52g": 5, "53g": 5, "54g": 5, "94": 5, "55g": 5, "56g": 5, "57g": 5, "96": 5, "58g": 5, "59g": 5, "97": 5, "60g": 5, "61g": 5, "98": 5, "62g": 5, "99": 5, "63g": 5, "64g": 5, "29tb": 5, "untar": 5, "content": 5, "attempt": [5, 8, 11], "new": 5, "configur": 5, "json": 5, "project": [5, 8, 11, 12], "plot_ica_eeg": [5, 7], "deprecationwarn": 5, "now": 5, "pathlib": 5, "which": [5, 11, 12], "nativ": 5, "support": 5, "plu": 5, "oper": [5, 8], "switch": 5, "forward": 5, "slash": 5, "instead": 5, "remov": 5, "open": 5, "read": 5, "item": 5, "pca": 5, "102": 5, "idl": 5, "v2": 5, "v3": 5, "averag": 5, "refer": 5, "6450": 5, "48149": 5, "956": 5, "320": 5, "665": 5, "readi": 5, "41699": 5, "000": 5, "277": 5, "709": 5, "contigu": 5, "segment": 5, "set": [5, 8, 11], "up": 5, "band": 5, "hz": 5, "fir": 5, "paramet": [5, 8, 9, 10, 11], "design": 5, "one": 5, "zero": 5, "phase": 5, "non": [5, 8, 11], "causal": 5, "bandpass": 5, "window": 5, "domain": 5, "ham": 5, "0194": 5, "passband": 5, "rippl": 5, "db": 5, "stopband": 5, "attenu": 5, "lower": 5, "edg": 5, "transit": 5, "bandwidth": 5, "cutoff": 5, "frequenc": 5, "upper": 5, "length": 5, "497": 5, "310": 5, "after": 5, "reduc": 5, "ortho": [5, 6, 8, 11], "n_plot": 5, "t_plot": 5, "order": 5, "argsort": 5, "model": [5, 6, 8, 11], "observ": [5, 6], "recov": [5, 6], "figsiz": [5, 6], "ii": [5, 6], "set_titl": 5, "get_xaxi": 5, "set_vis": 5, "get_yaxi": 5, "offset": 5, "max": 5, "min": 5, "cumsum": 5, "newaxi": [5, 6], "tight_layout": 5, "078": [5, 7], "francoi": [6, 12], "independ": [6, 8, 11, 12], "analysi": [6, 8, 11, 12], "hessian": [6, 8, 11, 12], "approxim": [6, 8, 11, 12], "ieee": [6, 12], "transact": [6, 12], "signal": [6, 12], "1706": [6, 12], "08171": [6, 12], "seed": [6, 11], "2000": 6, "s1": 6, "sin": 6, "s2": 6, "s3": 6, "c_": 6, "std": 6, "standard": [6, 8, 11], "arrai": [6, 11], "comput": [6, 9, 12], "_": [6, 8, 12], "y_picard": 6, "y_picardo": 6, "red": 6, "steelblu": 6, "sharex": 6, "sharei": 6, "suptitl": 6, "sig": 6, "651": [6, 7], "plot_ica_synth": [6, 7], "184": 7, "execut": 7, "auto_exampl": 7, "mb": 7, "extend": [8, 11], "w_init": [8, 11], "m": [8, 11], "ls_tri": [8, 11], "lambda_min": [8, 11], "veri": 8, "fast": [8, 12], "int": [8, 11], "If": [8, 10, 11, 12], "bool": [8, 11], "option": [8, 10, 11], "enforc": 8, "otherwis": [8, 11, 12], "sub": [8, 11], "super": [8, 11], "gaussian": [8, 11], "With": 8, "recommend": [8, 12], "you": [8, 11, 12], "keep": 8, "see": [8, 11], "note": 8, "below": [8, 11], "perform": [8, 11], "either": [8, 11], "built": [8, 11], "exp": [8, 11], "cube": [8, 11], "should": [8, 11, 12], "contain": [8, 11], "two": [8, 9, 11], "call": [8, 11], "dure": [8, 11], "float": [8, 9, 11], "updat": 8, "ndarrai": [8, 9, 10], "l": [8, 11], "bfg": [8, 11], "memori": [8, 11], "backtrack": [8, 11], "line": [8, 11, 12], "search": [8, 11], "threshold": [8, 11], "eigenvalu": [8, 11], "ani": [8, 11, 12], "shift": [8, 11], "instanc": [8, 11], "when": [8, 9, 11], "specifi": 8, "normal": 8, "distribut": [8, 12], "reproduc": 8, "multipl": 8, "than": [8, 11], "mai": [8, 11], "lead": [8, 11], "errat": [8, 11], "linear": [8, 11], "henc": [8, 11], "domin": [8, 11], "further": [8, 11], "requir": [8, 11], "met": [8, 11], "load_digit": [8, 12], "return_x_i": [8, 12], "transform": [8, 12], "x_transform": [8, 12], "fit_transform": [8, 12], "1797": 8, "attribut": 8, "components_": 8, "appli": 8, "get": [8, 10, 11, 12], "equal": [8, 11], "unmixing_matrix": 8, "whitening_": 8, "mixing_": 8, "pseudo": 8, "invers": 8, "It": [8, 9, 11, 12], "map": 8, "mean_": 8, "over": [8, 11], "featur": [8, 11], "onli": [8, 11], "pre": [8, 11], "onto": [8, 11], "princip": [8, 11], "__init__": 8, "amari": 9, "distanc": 9, "matric": 9, "cancel": 9, "wa": 9, "scale": [9, 10], "input": 9, "diagon": 10, "close": 10, "boolean": [10, 11], "wrt": 10, "return_x_mean": 11, "return_n_it": 11, "check_fun": 11, "fastica_it": 11, "train": 11, "vector": 11, "extract": 11, "reduct": 11, "By": 11, "assum": 11, "have": [11, 12], "been": 11, "white": 11, "incorrect": 11, "In": 11, "case": 11, "ignor": 11, "x_mean": 11, "too": 11, "whether": 11, "correct": 11, "posit": 11, "scalar": 11, "give": [11, 12], "un": 11, "optionn": 11, "check": 11, "provid": [11, 12], "user": [11, 12], "begin": 11, "safe": 11, "comp": 11, "rotat": 11, "befor": 11, "might": 11, "better": 11, "point": 11, "inform": 11, "about": 11, "state": 11, "can": [11, 12], "obtain": 11, "inv": 11, "n_iter": 11, "taken": 11, "librari": 12, "real": 12, "These": 12, "even": 12, "perfectli": 12, "hold": 12, "anaconda": 12, "forg": 12, "need": 12, "add": 12, "your": 12, "channel": 12, "numexpr": 12, "Then": 12, "admin": 12, "privileg": 12, "flag": 12, "upgrad": 12, "everyth": 12, "work": 12, "fine": 12, "c": 12, "error": 12, "messag": 12, "easiest": 12, "wai": 12, "follow": 12, "output": 12, "introduc": 12, "mimic": 12, "pleas": 12, "github": 12, "issu": 12, "tracker": 12, "document": 12}, "objects": {"picard": [[8, 0, 1, "", "Picard"], [9, 2, 1, "", "amari_distance"], [10, 2, 1, "", "permute"], [11, 2, 1, "", "picard"]], "picard.Picard": [[8, 1, 1, "", "__init__"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "function", "Python function"]}, "titleterms": {"api": [0, 12], "document": 0, "precondit": [0, 5], "ica": [0, 5], "real": 0, "data": [0, 4], "exampl": 1, "galleri": 1, "us": [2, 5, 6], "custom": 2, "densiti": 2, "picard": [2, 3, 4, 6, 8, 9, 10, 11, 12], "compar": 3, "fastica": [3, 4], "task": 3, "detect": 3, "ecg": 3, "artifact": 3, "meg": 3, "comparison": 4, "o": [4, 6], "face": 4, "blind": [5, 6], "sourc": [5, 6], "separ": [5, 6], "eeg": 5, "comput": 7, "time": 7, "amari_dist": 9, "permut": 10, "instal": 12, "conda": 12, "pip": 12, "check": 12, "quickstart": 12, "new": 12, "0": 12, "6": 12, "scikit": 12, "learn": 12, "compat": 12, "depend": 12, "cite": 12, "bug": 12, "report": 12}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 56}})