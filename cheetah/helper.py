
import numpy as np
from sklearn.utils import resample

def bootstrap_mean_CI(data_samples, alpha=0.95, n_iterations=1000):
    '''Calculate the bootstrapped mean applying the percentile method to
       estimate a confidence interval (Efron method)'''
    
    stats = []
    for i in range(0, n_iterations):
        # Resample original samples
        data_resample = resample(data_samples, replace=True,
                                 n_samples=len(data_samples))
        # Calculate mean
        stats.append(np.mean(data_resample))
    # Confidence interval
    p_lower = ((1.0-alpha)/2.0) * 100
    p_upper = (alpha+((1.0-alpha)/2.0)) * 100
    lower_bound = max(0.0, np.percentile(stats, p_lower))
    upper_bound = min(1.0, np.percentile(stats, p_upper))
    # Print result
    print([lower_bound, upper_bound, (lower_bound+upper_bound)/2.0, 
          (upper_bound-lower_bound)/2])
