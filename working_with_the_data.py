import numpy as np # Needed for create_dataset to work
from utils import find_name_file_and_label

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
