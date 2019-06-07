
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances, ERPCovariances
from tqdm import tqdm

import sys
sys.path.append('.')
from braininvaders2015a.dataset import BrainInvaders2015a


from scipy.io import loadmat
import numpy as np
import mne

from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

scr = {}
dataset = BrainInvaders2015a()

# note that subject 31 at session 3 has a few samples which are 'nan'
# to avoid this problem I dropped the epochs having this condition

#load data
for subject in dataset.subject_list:

	sessions = dataset._get_single_subject_data(subject)
	scr[subject] = {}

	for session in sessions.keys():

		raw = sessions[session]['run_1']		

		# filter data and resample
		fmin = 1
		fmax = 24
		raw.filter(fmin, fmax, verbose=False)

		# detect the events and cut the signal into epochs
		events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
		event_id = {'NonTarget': 1, 'Target': 2}
		epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=0.8, baseline=None, verbose=False, preload=True)
		epochs.pick_types(eeg=True)

		# get trials and labels
		X = epochs.get_data()
		y = epochs.events[:,-1]
		y = LabelEncoder().fit_transform(y)

		# cross validation
		skf = StratifiedKFold(n_splits=5)
		clf = make_pipeline(XdawnCovariances(estimator='lwf', classes=[1], xdawn_estimator='lwf'), MDM())
		scr[subject][session] = cross_val_score(clf, X, y, cv=skf, scoring = 'roc_auc').mean()

		# print results of classification
		print('subject', subject)
		print('mean AUC :', scr[subject])

filename = './classification_scores.pkl'
joblib.dump(scr, filename)



