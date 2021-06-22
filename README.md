# Thesis

This is the algorithm used to run the tests explained in my work of thesis.
The algorithm is developed on Python 3.7.10

In the file named How_to_use.py can be found a guided example on how to use the functions provided in the file named working_with_the_data.py. 
The remaining files are needed for working_with_the_data.py (and consequently also How_to_use.py) to work. 
They are organized in utils.py and corrected_ttest.py:

  - utils.py includes useful functions:
    - download_edf() strictly depends on the chosen dataset and downloads its files in the edf format; the chosen dataset can be found at the following link https://physionet.org/content/eegmmidb/1.0.0/
    - find_name_file_and_label() strictly depends on the chosen dataset and returns the name of the file and assigns a label to it; needed to read the edf file and to create the features matrix to perform the pattern recognition's task
    - create_KNN() creates matrices to store the accuracies obtained in the different classification tests
  - corrected_ttest.py includes the functions needed to perform the t-tests:
      - dependent_ttest() uses the most generic formula
      - dependent_ttest_kfold() conducts the two-tailed t-test using a simpler formula available when the protocol used is the k-folds cross-validation
      - dependent_ttest_kfold_one_tail() performs the one-tailed t-test using a simpler formula available when the protocol used is the k-folds cross-validation
