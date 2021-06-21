import numpy as np # Needed for create_dataset to work
from scipy import signal as sig
from scipy import stats as st
import mne

from utils import find_name_file_and_label # Path might be different

def create_dataset(conditions, subject_start, subject_end, cfg, f_band, check_nperseg, bands, pli):
  """

  Function to create the initial dataset to use in the task of pattern recognition.

  INPUT
    conditions: integer, total conditions of interest (depends on the chosen dataset)
    subject_start: integer, first subject in the dataset (included, depends on the chosen dataset)
    subject_end: integer, last subject excluded (depends on the chosen dataset)
    cfg: dictionary with the following key-value pairs
          freqRange    - list with the frequency range used to compute the power spectrum (or the PLI) by ScorEpochs (see scipy.stats.spearmanr()
                          function)
          fs           - integer representing sample frequency
          windowL      - integer representing the window length (in seconds)
          smoothFactor - smoothing factor for the power spectrum (0 by default, only available if the PSD is computed)
          wOverlap     - integer representing the number of seconds of overlap between two consecutive epochs (0 by
                          default, only available if the PSD is computed)
    f_band: list with the frequency range used to compute the PSD for task of pattern recognition (used only if the PSD is computed)
    check_nperseg: boolean, if True used to check whether the default nperseg parameter is usable to compute correctly the Spearman coefficient,
                  if not usable, nperseg will be changed. If False, the default value will be used without checking. Used only if the PSD is computed
    bands: dictionary of frequency bands of interest (used only if the PLI is computed)
    pli: boolean, True if the PLI is used instead of the PSD

  OUTPUT
    X_all: 2d list of features with shape: (subjects X epochs per subject X conditions, channels X features)
    y_all: array of labels of lenght: subjects X epochs_per_subject X conditions
    scores_all: array of scores computed by ScorEpochs of lenght: subjects X epochs_per_subject X conditions
    id_all: array of ids of the epochs for each subject and condition: subjects X epochs_per_subject X conditions
  """

  X_list = [] # Since the total number of subjects, epochs, channels and features is unknown (i.e. some subjects in the list cannot be used), a list is created.
  y_list = [] # Labels array
  scores_list = [] # Array of scores computed by ScorEpochs
  for condition in range(conditions): # In this dataset: EO -> condition == 0; EC -> condition == 1
    for subject in range(subject_start, subject_end):
      if subject not in [88, 92, 100]: # In this dataset these subjects have a different sample frequency from the others.
        print("The subject considered is {} and the condition is {} (0 -> EO and 1 -> EC)\n".format(subject, condition))
        file, label = find_name_file_and_label(subject, condition) # Provides the file and the label for the given subject and condition (depends on chosen dataset).
        data = mne.io.read_raw_edf(file) # Reads the edf file
        raw_data = data.get_data() # Extracts the EEG data matrix
        nCh = len(raw_data) # Computes the number of channels

        if pli: # Compute the PLI if desired and build X_list with it
          idx_best, epochs, scores = scorEpochs_PLI(cfg, raw_data, bands)
          """
          ScorEpochs provides to output:
          idx_best_ep: list of indexes sorted according to the best score (this list should be used for the selection of the
                      best epochs)
          epochs: 3d list of the data divided in equal length epochs of length windowL (epochs X channels X time samples)
          scores: array of scores per epoch
          """
          X_filtered = filter_data(raw_data, cfg["fs"], bands)                      # Filters the data to prepare it for the feature extraction
          eps, epoch_lenght = split_epoch(X_filtered, cfg["fs"], cfg["windowL"])    # Splits the data in epochs to prepare it for the feature extraction
          X_PLI = np.array ( [ PLI(np.transpose(e)) for e in eps ])                 # Computes the PLI for each epoch
          X_list.append(X_PLI)
        else: # Compute the PSD if desired and build X_list with it
          idx_best, epochs, scores = scorEpochs(cfg, raw_data, check_nperseg)
          """
          See also https://github.com/smlacava/scorepochs/blob/master/Python/scorEpochs.py,
          which is the same except for the check_nperseg input and the whole change of the nperseg parameter
          """
          psd = feature_extraction(cfg, f_band, nCh, epochs, check_nperseg)
          """
          feature_extraction returns the PSD computed in the frequency range given as f_band. The algorithm used to compute the PSD
          is exactly the same as the one ScorEpochs uses, except when check_nperseg = True.
          ScorEpochs uses the default value of nperseg, the feature extraction might use a changed value.
          See https://github.com/bpresicci/tesi/blob/main/feature_extraction.py for details.
          """
          X_list.append(psd)

        if label: # If label == 1, y of each epoch of that subject is 1, else 0. It depends on the chosen dataset.
          y_per_epoch = np.ones(len(epochs)) 
        else:
          y_per_epoch = np.zeros(len(epochs))
        
        y_list.append(y_per_epoch)
        scores_list.append(scores)
        if subject == subject_start and condition == 0: # An id is created for each epoch of each subject and condition: EO -> id>0, EC -> id<0.
          id_all = np.array(subject * np.ones(len(epochs)))
        elif condition == 0:
          id_all = np.append(id_all, np.array(subject * np.ones(len(epochs))))
        elif condition != 0:
          id_all = np.append(id_all, np.array(-subject * np.ones(len(epochs))))
      else:
        continue
  X_list = np.array(X_list)
  y_list = np.array(y_list)
  scores_list = np.array(scores_list)
  total_subjects = len(X_list)
  tot_epochs = len(X_list[0]) # Total epochs for each subject
  tot_channels = len(X_list[0][0])
  tot_samples = len(X_list[0][0][0])
  X_all = np.reshape(X_list, [total_subjects * tot_epochs, tot_channels * tot_samples])
  y_all = np.reshape(y_list, [total_subjects * tot_epochs])
  scores_all = np.reshape(scores_list, [total_subjects * tot_epochs])
  return X_all, y_all, scores_all, id_all



def scorEpochs_PLI(cfg, data, bands):
    """
    Function to select the best (most homogenoous) M/EEG epochs from a
    resting-state recordings.

    INPUT
       cfg: dictionary with the following key-value pairs
            freqRange    - list with the frequency range used to compute the PLI by ScorEpochs
            fs           - integer representing sample frequency
            windowL      - integer representing the window length (in seconds)

       data: 2d array with the time-series (channels X time samples)
       bands: dictionary of frequency bands of interest

    OUTPUT

       idx_best_ep: list of indexes sorted according to the best score (this list should be used for the selection of the
                     best epochs)

       epochs:       3d list of the data divided in equal length epochs of length windowL (epochs X channels X time samples)

       score_x_ep:   array of score per epoch
    """

    X = filter_data(data, cfg["fs"], bands)                                  # Perform the filtering of the data
    epochs, epoch_lenght = split_epoch(X, cfg["fs"], cfg["windowL"])         # Split the filtered data in epochs
    X_PLI = np.array([PLI(np.transpose(epoch)) for epoch in epochs])         # Compute the PLI for each epoch
    n_ch = len(data)
    n_ep = len(epochs)
    X_PLI_x_ch = np.zeros((n_ep, n_ch))
    score_ch_x_ep = np.zeros((n_ch, n_ep))
    for c in range(n_ch):
        for e in range(n_ep):
            X_PLI_x_ch[e] = X_PLI[e][c]
        score_ch, p = st.spearmanr(X_PLI_x_ch, axis=1)          # Correlation between the PLI of the epochs within each channel
        score_ch_x_ep[c][0:n_ep] += np.mean(score_ch, axis=1)   # Mean similarity score of an epoch with all the epochs for each channel
    score_x_ep = np.mean(score_ch_x_ep, axis=0)                 # The score of each epoch is equal to the mean of the scores of all the channels in that epoch
    idx_best_ep = np.argsort(score_x_ep)                        # Obtains of the indexes from the worst epoch to the best
    idx_best_ep = idx_best_ep[::-1]                             # Reversing to obtain the descending order (from the best to the worst)
    return idx_best_ep, epochs, score_x_ep
  
  def filter_data(raw_data, srate, bands):
    """
     Function that applies a filter to the frequencies.

     INPUT
       raw_data: 2d array with the time-series EEG data of size: number of channels X samples
       srate: integer, sampling rate
       bands: dictionary of frequency bands of interest

     OUTPUT
       filtered_data: 2d array with the filtered EEG data
    """
    for band in bands:
        low, high = bands[band]
        filtered_data = mne.filter.filter_data(raw_data, srate, low, high)
    return filtered_data

def split_epoch(X, srate, t_epoch_lenght, t_discard=0):
    """
    Function that divides the signal in epochs.
    INPUT
     X:  2d array with the time-series EEG data.  number of channels X samples
     srate: integer, sampling rate
     t_epoch_lenght: integer, lenght of the epoch (in seconds)
     t_discard: integer, initial portion of the record to be deleted (to eliminate initial artifacts)

    OUTPUT
     epochs: list of the data divided in equal length epochs
     epoch_lenght: integer, lenght in samples - number of samples (per channel) for each epoch
    """

    [n_channels, n_sample] = X.shape

    i_0 = t_discard * srate

    epoch_lenght = t_epoch_lenght * srate

    n_epochs = round((n_sample - i_0 - 1) / epoch_lenght)

    epochs = []
    for i_epoch in range(n_epochs):
        i_start = i_0 + i_epoch * epoch_lenght
        i_stop = i_start + epoch_lenght
        epoch = X[0:n_channels, i_start:i_stop]
        epochs.append(epoch)

    return epochs, epoch_lenght



def PLI(epoch):
    """
    Function to compute the PLI.
    INPUT
      epoch: 2d array with the data in the epoch
    OUTPUT
      PLI: 2d symmetric matrix, with shape number of channels X number of channels, containing the computed PLI values
    """
    nLoc = np.shape(epoch)[-1]
    PLI = np.zeros(shape=(nLoc, nLoc))
    complex_sig = sig.hilbert(epoch)
    for i in range(nLoc - 1):
        for j in range(i + 1, nLoc):
            PLI[i, j] = abs(np.mean(np.sign(np.angle(np.divide(complex_sig[:, i],complex_sig[:, j])))));
            PLI[j, i] = PLI[i, j];
    return PLI
