# Artifact-Detection-in-EEG-Signals
Authors: Andreas L. Jørgensen, Malene S. Thyrsted.

Supervisor: Kaare B. Mikkelsen.

The python code and notebooks for our bachelor project in Data Science.

## File Structure
The repo consist of a 

mne-methods/ (python package)
* __init__.py
* ICA_subfunctions.py
  * multi_plot(min_n_ica, max_n_ica, folder_path, matrix_list, cross_measure)
  * multi_ica(raw, min_n_ica, max_n_ica, folder_path, matrix, cross_measure)
  * M_max(A)
    * Input: matrix as np.array type
    * Output: matrix of max values for each channel.
    * Calculatethe M_max of the matrix, following the formula in bachelor.
  * M_idx(A)
  * artifact_classifier(folder_name, relative_procent, matrix, cross)
  * broken_channel_classifier(artifact_intervals, broken_interval_size)
  * ICA_and_Classifier_function(raw_proces, data_path, intv_start, relative_procent, intv_count, intv_size)
* load_mne.py
  * file_to_raw(data_path)
  * calculate_mean_channel(raw)
  * update_channels(raw, mean_channel)
  * preprocess(raw)
    * Input: raw file as mne.raw type
    * Output: mean referenced raw file, nans
    * Mean references, replacing NaN with zero and filter data

and three jupyter-notebooks:
* Load, plot, calculate IC from raw file
* Optimisation_Measured_data.ipynb
* Optimisation_Simulated_data.ipynb
