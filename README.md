# *US Accounting Analysis* - An analysis on data from a Reddit r/accounting survey

## Goal:
Because my wife was transitioning into the field of accounting and I was looking to develop projects to demonstrate proficiency with data analysis, I combined the two ideas together to create this project.  Additionally, she had done her own research on the field, stating that accountants with their CPA license earn more, as do accountants who start at a public accounting firm, then transition into working as an accountant in industry at a private company.  I used this data to better understand if those statements are true, as well as what a typical accountant looks like.

## Method:
Using Python, I broke the process down into three Jupyter notebooks:
1. Data cleaning
2. Exploratory Data Analysis
3. Regression Analysis


### Part 1 - Data Cleaning:
Using `Pandas`, `NumPy`, and `Matplotlib`, my main priorities were as follows:
* Remove unnecessary data that will not aid our analysis
  * Removed the time at which someone responded to the survey
* Focus on US Accountants.  The survey was global, however given that my wife and I live in the US, it didn't make sense to have accountants from other countries in the analysis.
  * 87% of the survey were accountants in the US, leaving 1201 accountants in total for the analysis.
* Working Column by column:
  * "Current Salary + Bonus"
    *  Entire Column was 

