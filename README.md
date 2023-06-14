# Artifact-Detection-in-EEG-Signals
Authors: Andreas L. Jørgensen, Malene S. Thyrsted.

Supervisor: Kaare B. Mikkelsen.

The python code and notebooks for our bachelor project in Data Science.

## File Structure
The repository consists of:

mne-methods/ (python package)
* __init__.py
* ICA_subfunctions.py
  * multi_plot(min_n_ica, max_n_ica, folder_path, matrix_list, cross_measure)
    * Input: minimum number of components in ICA, maximum number of components in ICA, folder path to load the ICA from, list of the matrices to plot default = ['um', 'mm'], cross measures to plot default = ['M_idx', M_max'].
    * Output: None.
    * Plots and saves the plots of the output on the cross measures of the matrices.
  * multi_ica(raw, min_n_ica, max_n_ica, folder_path, matrix, cross_measure)
  * M_max(A)
    * Input: matrix as np.array type.
    * Output: matrix of max values for each channel.
    * Calculates the M_max of the matrix, following the formula in the bachelor report.
  * M_idx(A)
    * Input: matrix as np.array type
    * Output: matrix of M_idx values for each channel.
    * Calculates the M_idx of the matrix, following the formula in the bachelor report. 
  * artifact_classifier(folder_name, relative_procent, matrix, cross)
    * Input: folder name to load the matrix and cross-measure from, relative percent to use as threshold, matrix default = 'mm', cross-measure default = 'M_max'.
    * Output: list of predicted artifacts and broken channels.
    * Calculated the predicted artifacts and broken channels of the matrix and cross-measure based on the relative percent.
  * broken_channel_classifier(artifact_intervals, broken_interval_size)
    * Input: list of list of predicted artifacts and broken channels for multiple intervals, number of consecutive intervals for predicting a channel as broken default = 3.
    * Output: list of predicted broken channels.
    * Runs through a list of list of predicted artifacts and broken channels, returns all channels that are broken in n = broken_interval_size intervals in a row.
  * ICA_and_Classifier_function(raw_proces, data_path, intv_start, relative_procent, intv_count, intv_size)
    * Input: preprocessed raw file as mne.raw type, data path to save computations in, timestamp to start the interval in, relative percent to use as threshold, number of consecutive intervals run on default = 6, size of intervals in seconds default = 10.
    * Output: list containing a list of list of predicted artifacts and broken channels for multiple intervals and a list of predicted broken channels.
    * Runs multiple_ica() and multi_plot(), then predicts broken channels using artifact_classifier() and broken_channel_classifier().
* load_mne.py
  * file_to_raw(data_path)
    * Input: data path of the raw EEG .mat file.
    * Output: raw file as mne.raw type.
    * Loads a .mat file containing the raw EEG signal, creates mne.info file containing info on sampling frequency and channel names returns it all as a mne.raw.
  * calculate_mean_channel(raw)
    * Input: raw file as mne.raw type.
    * Output: list of means per column.
    * Calculates the overall mean for all channels per column in the raw signal.
  * update_channels(raw, mean_channel)
    * Input: raw file as mne.raw type, list of means per column.
    * Output: None.
    * Updates the mne.raw file by subtracting the mean channel from calculate_mean_channel(), thus ensuring zero-mean for the raw file.
  * preprocess(raw)
    * Input: raw file as mne.raw type
    * Output: mean referenced raw file, nans
    * Mean references, replacing NaN with zero and filter data

and three jupyter-notebooks:
* Load, plot, calculate IC from raw file
* Optimisation_Measured_data.ipynb
* Optimisation_Simulated_data.ipynb
* Break_test_simulated.ipynb
