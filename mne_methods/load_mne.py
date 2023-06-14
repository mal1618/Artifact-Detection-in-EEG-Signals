import h5py
import mne
import numpy as np
from mne.preprocessing import (ICA)
import warnings

def file_to_raw(data_path):
    '''
    load .mat EEG_raw and convert to MNE raw
    '''
    # load data and srate:
    with h5py.File(data_path, 'r') as file:
        data = list(file['data'])
        srate = list(file['srate'])
    
    # channel labels:
    ch_names = ['ELA','ELB','ELC','ELT','ELE','ELI','ERA','ERB','ERC','ERT','ERE','ERI','EOGr',
            'EOGl','EMGl','EMGr','M1','F3','C3','O1','M2','F4','C4','O2','EMGc']
    ch_types = ['eeg'] * 25
    
    # creating info:
    sampling_freq = int(srate[0])
    info = mne.create_info(ch_names = ch_names, sfreq=sampling_freq, ch_types = ch_types)
    
    # transposing data
    data_t = np.array(data).T
    
    # scaling data from mikro volt to volt.
    data_t = data_t * 10**(-6)
    
    # creating raw
    raw = mne.io.RawArray(data_t, info, verbose = False)
    
    return raw

def calculate_mean_channel(raw):
    # 25 channels print(np.shape(raw[:][0]))
    mean_channel = np.nanmean(raw[:][0], axis=0)
    
    return mean_channel

def update_channels(raw, mean_channel):
    # Update the channels
    raw._data = raw._data - mean_channel
    
    return None

def preprocess(raw):
    '''
    Mean references and filters data taking into account possible NaN values.
    '''
    # save raw as array:
    r = raw.get_data()
    
    # find Nans, save index and replace with 0:
    all_nans = []
    for channel in range(len(r)):
        chan_nans = []
        for data_point in range(len(r[channel])):
            if np.isnan(r[channel,data_point]):
                chan_nans.append(data_point)
                # update r:
                r[channel,data_point] = 0
        all_nans.append(chan_nans)
        
    # transform back to mne raw file type:
    raw_process = raw.copy()
    raw_process._data = r
    
    # calculate mean and mean reference data:
    mean_channel = calculate_mean_channel(raw_process)
    update_channels(raw_process, mean_channel)
    
    # filtering data (without warning)
    mne.set_log_level(verbose=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Keyword argument 'nyq' is deprecated in favour of 'fs' and will be removed in SciPy 1.12.0")
        raw_process = raw_process.filter(l_freq=1., h_freq=100.)
        raw_process = raw_process.notch_filter(freqs = (50))
        raw_process = raw_process.notch_filter(freqs = (100))
    
    # reinstating NaN values:
    #for ch in range(len(all_nans)):
     #   if len(all_nans[ch]) > 0:
      #      for dp in all_nans[ch]:
       #          raw_process._data[ch, dp] = np.nan
    
    return raw_process, all_nans