import os # Needed for download_edf to work
import numpy as np # Needed for create_dataset to work

def download_edf(subject_start, subject_end, conditions):
  """
  Function to download edf files from a dataset.
  Be aware that here it is used the one provided by the
  following link: https://physionet.org/content/eegmmidb/1.0.0/,
  but any dataset can be used.
  Change the code if you want to use another dataset.

  INPUT

    subject_start: integer, first subject in the dataset (included)
    subject_end: integer, last subject excluded
    conditions: integer, total conditions of interest

  """

  for condition in range(1, conditions + 1): 
    # In this dataset: EO -> condition == 1; EC -> condition == 2, so condition has to be either 1 or 2.
    for subject in range(subject_start, subject_end):
        if subject not in [88, 92, 100]: # In this dataset these subjects have a different sample frequency from the others.
            if subject < 10:
                url = 'https://physionet.org/files/eegmmidb/1.0.0/S00{}/S00{}R0{}.edf'.format(subject, subject, condition)
            if subject > 9 and subject < 100:
                url = 'https://physionet.org/files/eegmmidb/1.0.0/S0{}/S0{}R0{}.edf'.format(subject, subject, condition)
            if subject > 99:
                url = 'https://physionet.org/files/eegmmidb/1.0.0/S{}/S{}R0{}.edf'.format(subject, subject, condition)
            # Once the url is found, the file is downloaded
            os.system('wget {}'.format(url)) # The wget command downloads the file of the given url
        else:
            continue

def find_name_file_and_label(subject, condition):
  """
  Provides the file and the label for the given subject and condition (depends on chosen dataset).
  INPUT:
    subject: integer, the subject of which edf file is desired
    condition: integer, the condition desired
  OUTPUT:
    file: string with the name of the edf file desired.
    label: integer with the label. EO -> 0, EC -> 1.
  """
  condition += 1 # condition is either 0 or 1, but 1 and 2 are needed.
  if condition == 1:
      label = 0
  if condition == 2:
      label = 1
  if subject < 10:
      file = 'S00{}R0{}.edf'.format(subject, condition)
  if subject > 9 and subject < 100:
      file = 'S0{}R0{}.edf'.format(subject, condition)
  if subject > 99:
      file = 'S{}R0{}.edf'.format(subject, condition)
  return file, label

def create_KNN(perc, names, kf, n_random):
    """
    It creates matrices to store the accuracies obtained in the different tests.
     
    INPUT
        perc: array of percentages of epochs kept to create the new dataset from the initial one (1 - percentage is the percentage of removed epochs)
        names: list of the names of the classifiers used
        kf: object created to perform the k-folds cross-validation
        n_random: integer, it defines how many tests will be run to estimate the performance when removing random epochs
        
    OUTPUT
       Matrices meant to store the accuracies obtained in the different tests:
        - "w" stands for "worst", the matrix will contain the accuracies obtained from the datasets created by removing the worst epochs.
        - "b" stands for "best", the matrix will contain the accuracies obtained from the datasets created by removing the best epochs.
        - "r" stands for "random", the matrix will contain the accuracies obtained from the datasets created by removing epochs randomly.
              They have different dimensions because they include all 10 tests, not only one unlike the other cases.
        - "ucs" stands for "user and condition specific", these matrices are meant to be used for that approach.
    """
    KNN_w_scores = np.zeros((len(perc), len(names), kf.get_n_splits()))
    KNN_b_scores = np.zeros((len(perc), len(names), kf.get_n_splits()))
    KNN_r_scores_all = np.zeros((len(perc), n_random, len(names), kf.get_n_splits()))
    KNN_ucs_w_scores = np.zeros((len(perc), len(names), kf.get_n_splits()))
    KNN_ucs_b_scores = np.zeros((len(perc), len(names), kf.get_n_splits()))
    KNN_ucs_r_scores_all = np.zeros((len(perc), n_random, len(names), kf.get_n_splits()))
    
    return KNN_w_scores, KNN_b_scores, KNN_r_scores_all, KNN_ucs_w_scores, KNN_ucs_b_scores, KNN_ucs_r_scores_all
