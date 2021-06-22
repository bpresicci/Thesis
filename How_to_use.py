# This is an example to show how to use the functions provided to run the tests.

!pip install mne #Needed to read edf files. This command works on Google Colab
import mne
import numpy as np

# Be aware that the path might be different.

from utils import download_edf, create_KNN
from working_with_the_data import create_dataset, apply filters

# from corrected_ttest import dependent_ttest_kfold Import this if the desired t-test is the two-tailed one.
from corrected_ttest import dependent_ttest_kfold_one_tail

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

alpha = 0.05                      # Used to perform the t-test
subject_start = 1                 # Integer, first subject in the dataset (included)
subject_end = 2                   # Integer, last subject excluded
conditions = 2                    # Integer, total conditions of interest
t_ep = 3                          # Integer, the window length (in seconds)
fs = 160                          # Integer, sample frequency of the recordings
freq = [8, 13]                    # Interval of frequencies used to compute the PSD during the feature extraction
freq_scorepochs = [1, 40]         # Interval of frequencies used to compute the PSD using ScorEpochs
check_nperseg = False             # Boolean, decides whether to use the default value of nperseg or to check if it needs to be changed
w_overlap = 0                     # Integer, overlap between two consecutive epochs (optional, in seconds)
cfg = {'freqRange':freq_scorepochs, 'fs':fs, 'windowL':t_ep, 'wOverlap': w_overlap} # Dictionary needed to use ScorEpochs
""" 
If it is desired to use ScorEpochs only in a specific band, make sure to set check_nperseg as True
and cfg['freqRange'] = freq, be freq the interested band of frequencies
"""
uc_specific = False               # Boolean, if True it uses the approach specific for user and condition, if False it uses the group level approach
n_random = 10                     # Integer, it defines how many tests will be run to estimate the performance when removing random epochs
perc = np.arange(0.5, 1.03, 0.05) # Array of percentages of epochs kept to create the new dataset from the initial one (1 - percentage is the percentage of removed epochs)
bands = {"Alpha": (8, 13)}        # Dictionary of frequency bands of interest to compute the PLI (Needed to be set even when the PSD is desired, could be an empty dictionary)
pli = False                       # Boolean, True if the PLI is used instead of the PSD
w = True                          # Boolean, True if it is desired to remove the worst epochs to create the new datasets, False otherwise
r = True                          # Boolean, True if it is desired to remove random epochs to create the new datasets, False otherwise
b = False                         # Boolean, True if it is desired to remove the best epochs to create the new datasets, False otherwise

# Downloads the edf files of the chosen dataset (has to be different if the chosen dataset is different)
download_edf(subject_start, subject_end, conditions)

# Creates the initial dataset
X_all, y_all, scores_all, id_all = create_dataset(conditions, subject_start, subject_end, cfg, freq, check_nperseg, bands, pli)

# Create object to perform the k-folds cross-validation (k = n_splits = 10)
kf = KFold(n_splits=10, shuffle = True, random_state = 42)

# Create the object to perform the classification by using the KNN algorithm 
names =["Nearest Neighbors 1","Nearest Neighbors 3","Nearest Neighbors 5"] 
classifiers =[    
              KNeighborsClassifier(1), 
              KNeighborsClassifier(3),
              KNeighborsClassifier(5)
              ]
              
"""
Initialize the matrices that will store the accuracies obtained from all tests:
        - "w" stands for "worst", the matrix will contain the accuracies obtained from the datasets created by removing the worst epochs.
        - "b" stands for "best", the matrix will contain the accuracies obtained from the datasets created by removing the best epochs.
        - "r" stands for "random", the matrix will contain the accuracies obtained from the datasets created by removing epochs randomly.
              They have different dimensions because they include all 10 tests, not only one unlike the other cases.
        - "ucs" stands for "user and condition specific", these matrices are meant to be used for that approach.
"""
KNN_w_scores, KNN_b_scores, KNN_r_scores_all, KNN_ucs_w_scores, KNN_ucs_b_scores, KNN_ucs_r_scores_all = create_KNN(perc, names, kf, n_random)

"""
For each percentage in the perc array a new dataset will be created and the classification will be performed, which accuracies will be stored in the
KNN_scores matrices just created. worst and random cannot be True at the same time, because the indeces are sorted based on them. In order to
perform the filtering multiple times with a given percentage, it is compulsory to make sure that worst and random are not both True. Based on the
inputs w (remove worst epochs), r (remove random epochs) and b (remove best epochs), worst and random can be properly set 
and the filtering can be performed more than once at a time.
"""
for percentage, idx_perc in zip(perc, range(len(perc))):
  print("The current percentage of epochs included in the new dataset is {}, uc_specific is set to {}\n".format(percentage, uc_specific))
  if uc_specific:
    if w:
      worst = True
      random = False
      KNN_ucs_w_scores[idx_perc] = apply_filters(names, classifiers, kf, n_random, percentage, uc_specific, worst, random, conditions, subject_start, subject_end, X_all, y_all, scores_all, id_all)
    if r:
      worst = False
      random = True
      KNN_ucs_r_scores_all[idx_perc] = apply_filters(names, classifiers, kf, n_random, percentage, uc_specific, worst, random, conditions, subject_start, subject_end, X_all, y_all, scores_all, id_all)
    if b:
      worst = False
      random = False
      KNN_ucs_b_scores[idx_perc] = apply_filters(names, classifiers, kf, n_random, percentage, uc_specific, worst, random, conditions, subject_start, subject_end, X_all, y_all, scores_all, id_all)
  else:
    if w:
      worst = True
      random = False
      KNN_w_scores[idx_perc] = apply_filters(names, classifiers, kf, n_random, percentage, uc_specific, worst, random, conditions, subject_start, subject_end, X_all, y_all, scores_all, id_all)
    if r:
      worst = False
      random = True
      KNN_r_scores_all[idx_perc] = apply_filters(names, classifiers, kf, n_random, percentage, uc_specific, worst, random, conditions, subject_start, subject_end, X_all, y_all, scores_all, id_all)
    if b:
      worst = False
      random = False
      KNN_b_scores[idx_perc] = apply_filters(names, classifiers, kf, n_random, percentage, uc_specific, worst, random, conditions, subject_start, subject_end, X_all, y_all, scores_all, id_all)

"""
To perform the t-test, the matrices storing the t and p values are initialised, so that each test run can be kept track.
len(KNN_r_scores_all[0]) = n_random so for each random test a t and p value can be stored
len(names) = amount of classifiers used
"""
t_value = np.zeros((len(KNN_r_scores_all[0]), len(names)))
p_value = np.zeros((len(KNN_r_scores_all[0]), len(names)))

for percentage, idx_perc in zip(perc, range(len(perc))):

  # As in Kuncheva's examples, the errors are used to run the t-tests
  error_w = np.around((1 - KNN_w_scores) * 100, 2)
  error_r_all = np.around((1 - KNN_r_scores_all) * 100, 2)

  for idx_clf in range(len(names)):
    for i in range(len(KNN_r_scores_all[0])):
      # For each classifier, each percentage and each random run, a t-test is performed and the results are stored
      t_value[i][idx_clf], df, cv, p_value[i][idx_clf] = dependent_ttest_kfold_one_tail(error_w[idx_perc][idx_clf], error_r_all[idx_perc][i][idx_clf], kf.get_n_splits(), alpha) 
      """
      Use dependent_ttest_kfold if the wanted t-test is the two tails one:
      t_value[i][idx_clf], df, cv, p_value[i][idx_clf] = dependent_ttest_kfold(error_w[idx_perc][idx_clf], error_r_all[idx_perc][i][idx_clf], kf.get_n_splits(), alpha) 
      """
