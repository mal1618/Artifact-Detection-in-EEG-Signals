import h5py
import mne
import numpy as np
import warnings
import timeit
import matplotlib.pyplot as plt
from mne.preprocessing import (ICA)
import os

ch_names = ['ELA','ELB','ELC','ELT','ELE','ELI','ERA','ERB','ERC','ERT','ERE','ERI','EOGr',
            'EOGl','EMGl','EMGr','M1','F3','C3','O1','M2','F4','C4','O2','EMGc']

def multi_plot(min_n_ica, max_n_ica, folder_path, matrix_list=['um', 'mm'], cross_measure=['M_idx', 'M_max']):
    """
    Makes a plot for M_idx all the channels in channels_list and #IC.     
    Input:
    min_n_ica = min number of ica
    max_n_ica = max number of ica
    channels_list = list of channels to plot.
    """
    channels_list = [x for x in range(25)]
    # Check if the folder is empty
    if not os.path.exists(folder_path):
        raise ValueError('folder_path don exist')
    folder_path = folder_path + "\IC"
    
    # Loop over matrix
    for matrix in matrix_list: 
        
        # Loop over cross measure
        for cross in cross_measure:
            
            number_of_ic = []
            M_idx = []
            # Loop to load all data.
            for ic in range(min_n_ica, max_n_ica):
                number_of_ic.append(ic)
                file_path = folder_path + str(ic) + cross + "_" + matrix +".npy"
                M_idx.append(np.load(file_path))
            
            # Add the correct channel to the data and plot it
            M_idx = np.array(M_idx)
            for chn in range(len(channels_list)):
                linestyle = 'solid'
                if (chn > 19):
                    linestyle = 'dotted'
                elif (chn > 9):
                    linestyle = 'dashed'
                plt.plot(number_of_ic, M_idx[:,chn], label=ch_names[channels_list[chn]], linestyle=linestyle)
            
            #Save complete M_idx
            file_path = folder_path + "_" + cross + "_" + matrix +".npy"
            np.save(file_path, M_idx)
            
            # Save plot of M_idx vs #IC.
            file_path = folder_path + "plot_" + cross + "_" + matrix
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1.2))
            plt.title(cross + "_" + matrix + " against #IC")
            plt.xlabel("#IC")
            plt.ylabel(cross)
            plt.grid(True)
            plt.savefig(file_path, bbox_inches="tight")
            #plt.show()
            plt.clf()
            
    return None    
    
        
        
def multiple_ica(raw, min_n_ica, max_n_ica, folder_path, matrix=['mm','um'], cross_measure=["M_idx", "M_max"]):
    """
    Performs multiple ICA on the raw file.
    Then calculates M_idx and save plots of unmixing matric + plots of IC.
    Input:
    raw = raw MNE file
    min_n_ica = min number of ica
    max_n_ica = max number of ica
    """
    start = timeit.default_timer()
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = folder_path + "\IC"
    
    # Loops through correct number of ica.
    for ic in range(min_n_ica, max_n_ica):
        # Perform ICA
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FastICA did not converge. Consider increasing tolerance or the maximum number of iterations")
            ica_components = ICA(n_components=ic, max_iter='auto', method='fastica', random_state=97, verbose=False)
        # ConvergenceWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.  warnings.warn(
            ica_components.fit(raw, verbose=False)
        
        """
        # Save IC
        file_path = folder_path + str(ic) + "_multiple-ica.fif"
        ica_components.save(file_path, verbose=False)
        
        # Save IC plot
        file_path = folder_path + str(ic) + "plot_ic"
        mne.viz.set_browser_backend('matplotlib', verbose=False)
        fig = ica_components.plot_sources(raw, show_scrollbars=False, title="IC" + str(ic), show=False)
        fig.savefig(file_path)
        plt.close('all')
        plt.clf()
        """
        if "mm" in matrix:
            # Calculates Mixing matrix
            mm = np.dot(ica_components.mixing_matrix_, ica_components.pca_components_[:ica_components.n_components_])
        
            # Save Mixing matrix
            file_path = folder_path + str(ic) + "mm"
            np.save(file_path, mm)
        
        
            # Save plot of Mixing matrix
            file_path = folder_path + str(ic) + "plot_mm"
            plt.imshow(mm, cmap='plasma')
            plt.grid(True)
            plt.colorbar()
            plt.savefig(file_path)
            plt.clf()
        
        if "um" in matrix:
            # Calculates Unmxing matrix
            um = np.dot(ica_components.unmixing_matrix_, ica_components.pca_components_[:ica_components.n_components_])
        
            # Save Unmixing matrix
            file_path = folder_path + str(ic) + "um"
            np.save(file_path, um)
        
            # Save plot of Unmixing matrix
            file_path = folder_path + str(ic) + "plot_um"
            plt.imshow(um, cmap='plasma')
            plt.grid(True)
            plt.colorbar()
            plt.savefig(file_path)
            plt.clf()
               
        if "M_idx" in cross_measure:
            if "um" in matrix:
                # Save M_idx values of Unmixing matrix
                file_path = folder_path + str(ic) + "M_idx_um"
                np.save(file_path, M_idx(um))
           
            if "mm" in matrix:
                # Save M_idx values of Mixing matrix
                file_path = folder_path + str(ic) + "M_idx_mm"
                np.save(file_path, M_idx(mm))

        if "M_max" in cross_measure:
            if "um" in matrix:
                # Save M_max values of Unmixing matrix
                file_path = folder_path + str(ic) + "M_max_um"
                np.save(file_path, M_max(um))
            
            if "mm" in matrix:
                # Save M_max values of Mixing matrix
                file_path = folder_path + str(ic) + "M_max_mm"
                np.save(file_path, M_max(mm))
                
    
    plt.close('all')
    plt.clf()
    mne.viz.set_browser_backend('qt')
    stop = timeit.default_timer()
    # print("Multiple ICA finished in " + str(stop - start) + "seconds")
    # print("" + folder_path)
    return None



def M_max(A):
    return np.max(np.abs(A),axis=0)

def M_idx(A):
    i_len = len(A)
    j_len = len(A[0])
    M = np.zeros((i_len, j_len))
    for j in range(j_len): #Channels
        for i in range(i_len): #IC
            row_column = np.append(A[:,j], A[i,:])
            M[i][j] = np.abs(A[i][j]) / (np.std(np.abs(row_column)))
    return np.max(M,axis=0)

def artifact_classifier(folder_name, relative_procent, matrix='mm', cross='M_max'):
    """
    Calculate the artifact(s)/Independent component(s) that is seperated from the rest of components.
    """
    channels_list = [x for x in range(25)]
    
    # Load complete M_idx
    file_path = folder_name + "\IC_" + cross + "_" + matrix +".npy"
    M_idx = np.load(file_path)
    n_comp, n_chn = M_idx.shape
    
    # Mean of the M_max of mixing matrix
    mean = np.mean(M_idx)
    # Mean of each channel
    ch_mean = np.mean(M_idx, axis=0)
    
    
    
    # Run through each channel and check if the are seperated from the mean.
    art_list = []
    for ch in range(n_chn):
        if (ch_mean[ch] > mean*relative_procent):
            art_list.append(ch_names[channels_list[ch]])
    
    return art_list

def broken_channel_classifier(artifact_intervals, broken_interval_size = 3):
    """
    Returns all channels that are broken in a given number of consecutive intervals. Default broken_interval_size is 3
    """
    # counts the interval the broken channels occurs in
    i = 0
    res = dict()
    for sub in artifact_intervals:
        for chan in range(len(sub)):
            if sub[chan] not in res:
                res[sub[chan]] = list()
            intervals = res[sub[chan]]
            intervals.append(i)
            res[sub[chan]] = intervals
        i += 1
    
    broken_channels = []
    
    # counts consecutive occurences
    for key in res.keys():
        lst = res[key]
        consec = [1]
        for x, y in zip(lst, lst[1:]):
            if x == y - 1:
                consec[-1] += 1
            else:
                consec.append(1)
        # checks consecutive occurences against broken_interval_size
        if max(consec) >= broken_interval_size:
            broken_channels.append(key)
        
    return broken_channels

def ICA_and_Classifier_function(raw_proces, data_path, intv_start, relative_procent, intv_count=6, intv_size=10):
    """
    Run in the intervals fra intv_start [0-10] [10-20] [20-30] [30-40] [40-50] [50-60]
    Starts performing multiple ICA
    Then it classify the 10 sec. interval.
    """
    
    # Remove a lot of output written out.
    intv_end = intv_start + intv_count * intv_size + 1 #Etc. 6*10 + 1
    time_intv = [x for x in range(intv_start, intv_end, intv_size)]
    
    art_list = []
    for i in range(intv_count):    
        # Load the interval
        raw_filt = raw_proces.copy().crop(tmin=time_intv[i], tmax=time_intv[i + 1]).pick_types(eeg=True)
        
        # Check if nan is missing
        channels = [x for x in range(25)]
        """
        # Check if works for multiple channels that are removed.
        ch_name_remove = ""
        if np.any(np.isnan(raw_filt._data[:25])):
            ch_to_remove = []
            for ch_idx in range(25):
                if np.any(np.isnan(raw_filt._data[ch_idx])):
                    ch_to_remove.append(ch_idx)
            for idx in ch_to_remove:
                channels.remove(idx)
                raw_filt.drop_channels(ch_names[idx])
                ch_name_remove += "_" + ch_names[idx].strip() # strip remove space in start and end of string
        """
        # Automatic write the folder name
        # data_path = r"D:\Bachelor-Project-Data\Kaare\study_1A_mat_simple\S_11\night_1\EEG_raw.mat"
        folder_name = str(data_path[:-4]) + "_" + str(intv_size) + "s_t="
        folder_name = folder_name + str(time_intv[i]) + "-" + str(time_intv[i + 1]) + "_" + str(len(channels)) + "chs"
        
        # Running multiple ICA with 2-15 numbers of components
        multiple_ica(raw_filt, 2, 15, folder_name, [time_intv[i], time_intv[i + 1]])
        multi_plot(2, 15, folder_name, channels)
        
        # Running 10 sec artifact classifier
        art_list.append(artifact_classifier(2, 15, folder_name, relative_procent, channels))
    
    # call the Broken Channel classifier
    broken_channels = broken_channel_classifier(art_list)
        
    return [art_list, broken_channels]

