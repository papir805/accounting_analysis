
def get_test_stat(x, function):
    """
    Returns the mean of a Pandas Series
    """
    if function == 'mean':
        return x.mean()
    elif function == 'median':
        return x.median()
        
def get_bootstrap(df, label, random_state=None):
    df_copy = df.copy(deep=True)
    sample_size = sum(df_copy[label].isnull()==False)
    new_sample = df_copy[label].sample(sample_size, replace=True, random_state=random_state)
    return new_sample

def run_one_trial(df, label, function):
    new_sample = get_bootstrap(df, label)
    new_test_stat = get_test_stat(new_sample, function)
    return new_test_stat

def conf_plotter(test_stats, label, num_repetitions, sample_size, function, left, right):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(test_stats, edgecolor='black')
    ax.axvline(left, color='red', linestyle='--')
    ax.axvline(right, color='red', linestyle='--')
    plt.title('95% Confidence Interval - Observed Test Statistic (n={})'.format(sample_size))
    plt.xlabel('{} ({}) - Num Repetitions: {}'.format(label, function, num_repetitions))
    plt.ylabel('Count')
    plt.show()
    
    
def get_conf_int(df, label, num_repetitions, function, plot=False):
    import numpy as np
    sample_stats = []
    for _ in range(num_repetitions):
        sample_stats.append(run_one_trial(df, label, function))
    left = np.percentile(sample_stats, 2.5)
    right = np.percentile(sample_stats, 97.5)
    if plot == True:
        sample_size = sum(df[label].isnull()==False)
        conf_plotter(sample_stats, label, num_repetitions, sample_size, function, left, right)
    return (left, right)


