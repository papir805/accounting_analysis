def calc_vif(X):
    """
    X: A pandas DataFrame object of numerical independent variables to be used in regression,
    Calculates the variance inflation factor of each independent variable in X
    against all of the other independent variables in X"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import pandas as pd
    vif = pd.DataFrame()
    vif['Variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif.sort_values('VIF', ascending=False)
    
    return(vif)


def create_scipy_linear_model(df, X_label, y_label, plot=False):
    import scipy.stats
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    
    new_model = scipy.stats.linregress(df[X_label], df[y_label])
    slope = new_model.slope
    intercept = new_model.intercept
    r_value = new_model.rvalue
    
    predictions = slope * df[X_label] + intercept
    residuals = df[y_label] - predictions
    sample_size = df.shape[0]
    rmse = np.sqrt(mean_squared_error(df[y_label], predictions))
    
    if plot == True:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,8))
        
        ax[0].scatter(df[X_label], df[y_label], label='Actual')
        ax[0].set_xlabel(X_label)
        ax[0].set_ylabel(y_label)
        ax[0].set_title('Scatter plot versus predicted values from Linear Regression (r = {:.4f}; n = {})'.format(r_value, sample_size))
        ax[0].plot(df[X_label], predictions, color='r', label='Predicted')
        ax[0].legend()
        
        ax[1].scatter(df[X_label], residuals)
        ax[1].axhline(0, color='r', linestyle='--')
        ax[1].set_xlabel(X_label)
        ax[1].set_ylabel('Error in {}'.format(y_label))
        ax[1].set_title('Residual Plot')
        
        plt.tight_layout()
        #plt.show()
        
    return new_model, predictions, residuals, rmse



def get_error_plots(residuals):
    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    sns.histplot(residuals, kde=True, ax=ax[0])
    ax[0].set_title('Distribution of Residuals')
    ax[0].set_xlabel('Residual (Actual Salary - Predicted Salary)')
    ax[0].set_xlim(-150_000, 300_001)

    stats.probplot(residuals, plot = ax[1])

    plt.tight_layout()
    plt.show()
    
    
    

def create_train_test_scipy_model(df, X_label, y_label, random_state=None, plot=False, log=False):
    from sklearn.model_selection import train_test_split
    import numpy as np
    import scipy.stats
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[X_label], 
        df[y_label], 
        test_size=0.3, 
        random_state=random_state
    )

    sample_size = df.shape[0]
    new_model = scipy.stats.linregress(X_train, y_train)

    slope = new_model.slope
    intercept = new_model.intercept
    r_value = new_model.rvalue

    train_predictions = slope*X_train + intercept
    train_residuals = y_train - train_predictions

    test_predictions = slope*X_test + intercept
    test_residuals = y_test - test_predictions
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_r_val = np.sqrt(r2_score(y_test, test_predictions))
    if log == True:
        test_rmse = np.sqrt(mean_squared_error(10**y_test, 10**test_predictions))
        test_r_val = np.sqrt(r2_score(10**y_test, 10**test_predictions))
    #print('Testing Data - RMSE: {:.2f}'.format(test_rmse))
    #print("Testing Data - R-Value: {:.4f}".format(test_r_val)
    if plot == True:
        fig, ax = plt.subplots(2, 2, figsize=(10,10))
        ax[0][0].scatter(X_train, y_train)
        ax[0][0].plot(X_train, train_predictions, color='red')

        fig.suptitle('Train/Test Split (n = {})'.format(sample_size))
        ax[0][0].set_xlabel('Years Experience')
        ax[0][0].set_ylabel('Current Salary + Bonus')
        ax[0][0].set_title('Training Dataset (r: {:.4f}, n: {})'.format(r_value, len(X_train)))


        ax[0][1].scatter(X_test, y_test)
        ax[0][1].plot(X_test, test_predictions, color='red')
        ax[0][1].set_title('Training Dataset (n: {})'.format(len(X_test)))
        ax[0][1].set_xlabel('Years Experience')

        ax[1][0].scatter(X_train, train_residuals)
        ax[1][0].axhline(0, color='red', linestyle='--')
        ax[1][0].set_title('Training Residuals')
        ax[1][0].set_xlabel('Years Experience')
        ax[1][0].set_ylabel('Error in predicted Salary ($)')

        ax[1][1].scatter(X_test, test_residuals)
        ax[1][1].axhline(0, color='red', linestyle='--')
        ax[1][1].set_title('Test Residuals')
        ax[1][1].set_xlabel('Years Experience')

        plt.tight_layout()
        plt.show()
        
    return new_model, test_predictions, test_residuals, test_rmse, test_r_val


# def create_conf_int_plots(rmse_list, r_val_list, left=0.025, right=0.975):
#     sample_size = len(rmse_list)
#     rmse_left, rmse_right = np.quantile(rmse_list, left), np.quantile(rmse_list, right)
#     r_val_left, r_val_right = np.quantile(r_val_list, left), np.quantile(r_val_list, right)
#     interval_width = right-left

#     fig, ax = plt.subplots(1,2, sharey=True, figsize=(10,5))
#     ax[0].hist(rmse_list, edgecolor='black')
#     ax[0].set_title('RMSE')
#     ax[0].set_ylabel('Count')
#     ax[0].axvline(rmse_left, color='red', linestyle='--')
#     ax[0].axvline(rmse_right, color='red', linestyle='--')
#     ax[0].axvline(test_rmse_yrs_exp_under_15, color='cyan', linestyle='-.')

#     ax[1].hist(r_val_list, edgecolor='black')
#     ax[1].set_title('R-Value')
#     ax[1].axvline(r_val_left, color='red', linestyle='--')
#     ax[1].axvline(r_val_right, color='red', linestyle='--')
#     ax[1].axvline(test_r_val, color='fuchsia', linestyle='-.')

#     plt.suptitle('n = {}'.format(sample_size))
#     plt.tight_layout()
#     plt.show();

#     print('RMSE {}% Confidence Interval for {} trials: ({:.2f}, {:.2f})'.format(interval_width*100, num_trials, rmse_left, rmse_right))
#     print('R-Value {}% Confidencet Interval for {} trials: ({:.4f}, {:.4f})'.format(interval_width*100, num_trials, r_val_left, r_val_right))
    
    
    
def create_conf_ints(df, X_label, y_label, num_trials, base_rmse, base_r_val, c_level = 0.95, log=False, plot=False):
    import scipy.stats
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    rmse_list = []
    r_val_list = []

    
    for trial in np.arange(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(df[X_label], df[y_label], test_size=0.3)
        trial_model = scipy.stats.linregress(X_train, y_train)
        
        slope, intercept = trial_model.slope, trial_model.intercept
        
        test_predictions = slope * X_test + intercept
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        r_value = np.sqrt(r2_score(y_test, test_predictions))
        
        if log==True:
            untransformed_test_predictions = 10 ** test_predictions
            untransformed_y_test = 10 ** y_test
            rmse = np.sqrt(mean_squared_error(untransformed_y_test, untransformed_test_predictions))
            r_value = np.sqrt(r2_score(untransformed_y_test, untransformed_test_predictions))
            
        rmse_list.append(rmse)
        r_val_list.append(r_value)
    
    sample_size = y_test.shape[0]
    
    if plot==True:
            rmse_left, rmse_right, r_val_left, r_val_right = create_conf_int_plots(rmse_list, r_val_list, base_rmse, base_r_val, c_level, sample_size, log)
    return rmse_left, rmse_right, r_val_left, r_val_right

def create_conf_int_plots(rmse_list, r_val_list, base_rmse, base_r_val, c_level, sample_size, log=False):
    import numpy as np
    import matplotlib.pyplot as plt
    
    left = (1 - c_level) / 2
    right = 1 - left
    num_trials = len(rmse_list)
    rmse_left, rmse_right = np.quantile(rmse_list, left), np.quantile(rmse_list, right)
    r_val_left, r_val_right = np.quantile(r_val_list, left), np.quantile(r_val_list, right)
    
    fig, ax = plt.subplots(1,2, sharey=True, figsize=(10,5))
    ax[0].hist(rmse_list, edgecolor='black')
    ax[0].set_title('RMSE')
    ax[0].set_ylabel('Count')
    ax[0].axvline(rmse_left, color='red', linestyle='--')
    ax[0].axvline(rmse_right, color='red', linestyle='--')
    ax[0].axvline(base_rmse, color='cyan', linestyle='-.')

    ax[1].hist(r_val_list, edgecolor='black')
    ax[1].set_title('R-Value')
    ax[1].axvline(r_val_left, color='red', linestyle='--')
    ax[1].axvline(r_val_right, color='red', linestyle='--')
    ax[1].axvline(base_r_val, color='fuchsia', linestyle='-.')

    plt.suptitle('num_trials = {} - n = {}'.format(num_trials, sample_size))
    if log==True:
        plt.suptitle('Log transformed - (num_trials = {}; n = {})'.format(num_trials, sample_size))
    plt.tight_layout()
    plt.show()
    return rmse_left, rmse_right, r_val_left, r_val_right
