# -*- coding: UTF-8 -*-

import mne
import numpy as np
from braininvaders2015a import download as dl
import os
import glob
import zipfile
import yaml
from scipy.io import loadmat
from distutils.dir_util import copy_tree
import shutil

BI2015a_URL = 'https://zenodo.org/record/3266930/files/'

class BrainInvaders2015a():
    '''

    '''

    def __init__(self):

        self.subject_list = list(range(1, 44 + 1))

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)

        sessions = {}
        for file_path, session in zip(file_path_list, [1, 2, 3]):

            session_name = 'session_' + str(session)
            sessions[session_name] = {}
            run_name = 'run_1'

            chnames = ['Fp1',
                        'Fp2',
                        'AFz',
                        'F7',
                        'F3',
                        'F4',
                        'F8',
                        'FC5',
                        'FC1',
                        'FC2',
                        'FC6',
                        'T7',
                        'C3',
                        'Cz',
                        'C4',
                        'T8',
                        'CP5',
                        'CP1',
                        'CP2',
                        'CP6',
                        'P7',
                        'P3',
                        'Pz',
                        'P4',
                        'P8',
                        'PO7',
                        'O1',
                        'Oz',
                        'O2',
                        'PO8',
                        'PO9',
                        'PO10',
                        'STI 014']

            chtypes = ['eeg'] * 32 + ['stim']               

            D = loadmat(file_path)['DATA'].T
            S = D[1:33,:]
            stim = D[-2,:] + D[-1,:]
            X = np.concatenate([S, stim[None,:]])

            info = mne.create_info(ch_names=chnames, sfreq=512,
                                   ch_types=chtypes, montage='standard_1020',
                                   verbose=False)
            raw = mne.io.RawArray(data=X, info=info, verbose=False)

            sessions[session_name][run_name] = raw

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        # check if has the .zip
        url = BI2015a_URL + 'subject_' + str(subject).zfill(2) + '_mat.zip'
        path_zip = dl.data_path(url, 'BRAININVADERS2015A')
        path_folder = path_zip.strip('subject_' + str(subject).zfill(2) + '.zip')

        # check if has to unzip
        path_folder_subject = path_folder + 'subject_' + str(subject).zfill(2) + os.sep
        if not(os.path.isdir(path_folder_subject)):
            os.mkdir(path_folder_subject)
            print('unzip', path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder_subject)

        # filter the data regarding the experimental conditions
        subject_paths = []        
        for session in [1, 2, 3]:
            subject_paths.append(path_folder_subject + 'subject_' + str(subject).zfill(2) + '_session_' + str(session).zfill(2) + '.mat')

        return subject_paths
