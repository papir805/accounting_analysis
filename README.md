# *US Accounting Analysis* - An analysis on data from a Reddit r/accounting survey

## Goal: Understand the typical accountant
Because my wife was preparing to enter into the field of accounting and I was looking to develop projects to demonstrate proficiency with data analysis, I combined the two ideas together to create this project.  Additionally, she had done her own research on the field, stating that accountants with their CPA license earn more, as do accountants who start at a public accounting firm, then transition into working as an accountant in industry at a private company.  I used this data to better understand if those statements are true, as well as what a typical accountant looks like.

## Method:
Using Python, I broke the process down into three Jupyter notebooks:
1. Data cleaning
2. Exploratory Data Analysis
3. Regression Analysis


### Part 1 - Data Cleaning:
Using `Pandas`, `NumPy`, and `Matplotlib`, I performed the following tasks:
1. Remove unnecessary data.
2. Clean strings of unwanted characters.
3. Recast columns to their appropriate types.
4. Handle `Null` Values.

#### Remove Unnecessary data that won't aid analysis:
 * Ex: removed the time at which someone responded to the survey.
 * Remove Non US Accountants.  The survey was global, however given that my wife and I live in the US, it didn't make sense to have accountants from other  countries in the analysis.
   * 87% of the survey were accountants in the US, leaving 1201 accountants in total for the analysis.

#### Cleaning Strings:
 * "Current Salary + Bonus"
   * Values like "73k" needed to be changed to "73000"
   * Replace commas and dollar signs with empty strings
 * "Line of Service"
   * Standardized entries by distilled 36 repeated unique strings down to just 9:
     * Ex: Internal Audit, IT Audit -> Audit
     * Ex: Audit/Tax, Audit & tax, Tax and audit -> Tax and Audit
 * "Gender"
   * Standardized by distilling 2 repeated unique strings down to 1:
     * Ex: Prefer not to say or Other -> Undisclosed

 #### Converting columns to appropriate types 
 * Recast "Current Salary + Bonus" to `float`
 * "Years of Experience", etc. to `int`
 
 * Cleaning `Null` values
   * Gender: Replaced `Null` values with Undisclosed
   * Line of Service: Replaced `Null` values with Other
   * Current Industry: Replaced `Null` values with Other
   * Distribution of salaries was right-skewed, imputed `Null` values with the median.
   * Distribution of years of experience was right-skewed, imputed `Null` values with the median.
   * Cost of living: Imputed `Null` Values with the mode.

### Part 2 - Exploratory Data Analysis
