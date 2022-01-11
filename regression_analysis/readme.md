# Part 3 - Regression Analysis:
Libraries used: `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scipy`, `Sklearn`, `Statsmodels`

**Goal**:
To determine which variable(s) do the best job at predicting a US accountant's Current Salary + Bonus and use them to construct a model that's trained on training data.  This model should have good performance on metrics like Root Mean Squared Error (RMSE) or Pearson's Correlation Coefficient (r), when verified against testing data. 

## How to use:
[Click here](https://github.com/papir805/accounting_analysis/blob/main/regression_analysis/accounting_regression-for-github.ipynb) or [here](https://nbviewer.org/github/papir805/accounting_analysis/blob/main/regression_analysis/accounting_regression-for-github.ipynb) to see code I used to construct the regression models, as well as the thought process that went behind selecting what I considered to best the best variable(s) and model.

## Method:
1. Use Heatmaps to determine which variables are most likely to be the best predictor variables
2. Construct simple linear regression models and verify performance using RMSE and the r-value.
3. Check to see if the conditions necessary for a linear regression model are present such as:
    * Analyzing residual plots.
    * Analyzing QQ normal probability plots
4. Retrain models using training data and verify model performance on testing data using RMSE and r-value.
5. Construct Polynomial regression models, verify performance, retrain on training data and verify performance against testing data.
6. Construct multiple linear regression models, verify performance, retrain on training data and verify performance against testing data.

### Conclusion:
A multivariate linear regression model using Has CPA and Years Experience appeared to offer solid performance with minimal complexity.  Using training data, verified against testing data, a 95% confidence interval for the RMSE is between $15,675.62 and $18,942.88, with a R-value between 0.60 and 0.72.
 
