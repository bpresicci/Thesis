# Python implementation of the Nadeau and Bengio correction of dependent Student's t-test
# using the equation stated in https://www.cs.waikato.ac.nz/~eibe/pubs/bouckaert_and_frank.pdf

from scipy.stats import t
from math import sqrt
from statistics import stdev

"""
The functions share the same output, whereas the input slightly changes.
- dependent_ttest(data1, data2, n_training_folds, n_test_folds, alpha) 
    performs the t-test in the most general way, without the additional information about the protocol used to build training and testing sets
- dependent_ttest_kfold(data1, data2, k, alpha)
    performs the two-tailed t-test using a simpler formula available when the protocol used is the k-folds cross-validation
    the formula is explained by L. I. Kuncheva in the 1.4.2 paragraph of “Combining Pattern Classifiers: Methods and Algorithms” 
- dependent_ttest_kfold_one_tail(data1, data2, k, alpha)
    performs the one-tailed t-test using a simpler formula available when the protocol used is the k-folds cross-validation
    
COMMON INPUT
    data1: array 1 of data to compare
    data2: array 2 of data to compare
    alpha: alpha value used to interpret the p-values
OUTPUT
    t_stat: t-value
    df: degrees of freedom
    cv: critical value
    p: p-value
"""

def dependent_ttest(data1, data2, n_training_folds, n_test_folds, alpha): 
    """
    Other than the common inputs, there are two additional inputs:
    n_training_folds: the size of the training set
    n_test_folds: the size of the test set
    """
    n = len(data1) 
    differences = [(data1[i]-data2[i]) for i in range(n)] 
    sd = stdev(differences)
    divisor = 1 / n * sum(differences) 
    test_training_ratio = n_test_folds / n_training_folds 
    denominator = sqrt(1 / n + test_training_ratio) * sd
    t_stat = divisor / denominator
    # degrees of freedom
    df = n - 1
    #calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0 
    # return everything
    return t_stat, df, cv, p

def dependent_ttest_kfold(data1, data2, k, alpha):
    """
    Other than the common inputs, there is one additional input:
    k: integer, the number of folds used for the k-folds cross-validation
    """
    n = len(data1)
    differences = [(data1[i]-data2[i]) for i in range(n)] 
    sd = stdev(differences)
    divisor = 1 / n * sum(differences)
    denominator = sqrt((2 * k - 1) / (k * (k - 1))) * sd
    t_stat = divisor / denominator
    # degrees of freedom
    df = n - 1
    #calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = 2 * t.cdf(-(abs(divisor)/denominator), df)
    # return everything
    return t_stat, df, cv, p

def dependent_ttest_kfold_one_tail(data1, data2, k, alpha):
    """
    Other than the common inputs, there is one additional input:
    k: integer, the number of folds used for the k-folds cross-validation
    """
    n = len(data1) 
    differences = [(data1[i]-data2[i]) for i in range(n)] 
    sd = stdev(differences)
    divisor = 1 / n * sum(differences)
    denominator = sqrt((2 * k - 1) / (k * (k - 1))) * sd
    t_stat = divisor / denominator
    # degrees of freedom
    df = n - 1
    #calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = t.cdf(t_stat, df)
    # return everything
    return t_stat, df, cv, p
