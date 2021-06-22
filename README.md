# Thesis

This is the algorithm used to run the tests explained in my work of thesis.
The algorithm is developed on Python 3.7.10

In the file named How_to_use.py can be found a guided example on how to reproduce the tests. It uses the functions provided in the file named working_with_the_data.py, which needs the remaining files in the branch to work.
Consequently, it is very important that all the files included in this branch are downloaded and correctly imported in order for How_to_use.py to work properly. The path currently indicated in files might be different for each user, so change it if needed.

  - working_with_the_data.py includes the fundamental functions to run the tests (except the ones needed to perform the t-tests):
    - create_dataset() Creates the initial dataset;
    - apply_filters() Creates new datasets by removing a specified percentage of epochs from the initial dataset;
    - Other functions are included in the file, but these two are the ones that How_to_use.py will need to import to work.

  - utils.py includes useful functions:
    - download_edf() strictly depends on the chosen dataset and downloads its files in the edf format; the chosen dataset can be found at the following link https://physionet.org/content/eegmmidb/1.0.0/
    - create_KNN() creates matrices to store the accuracies obtained in the different classification tests
  - corrected_ttest.py includes the functions needed to perform the t-tests:
      - dependent_ttest() uses the most generic formula
      - dependent_ttest_kfold() conducts the two-tailed t-test using a simpler formula available when the protocol used is the k-folds cross-validation
      - dependent_ttest_kfold_one_tail() performs the one-tailed t-test using a simpler formula available when the protocol used is the k-folds cross-validation
