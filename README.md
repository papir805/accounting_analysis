# **US Accounting Analysis** - An analysis on data from a Reddit r/accounting survey

## Overall goal - To understand the typical accountant and the accounting field:
Because my wife was preparing to enter into the field of accounting and I was looking to develop projects to demonstrate proficiency with data analysis, I combined the two ideas together to create this project.  Additionally, she had done her own research on the field, stating that accountants with their CPA license earn more, as do accountants who start at a public accounting firm, then transition into working as an accountant in industry at a private company.  I used this data to better understand if those statements are true, as well as what a typical accountant looks like.

## Method:
Using Python, I broke the process down into three Jupyter notebooks:
1. Data cleaning [Click here to see](https://github.com/papir805/accounting_analysis/tree/main/cleaning)
2. Exploratory Data Analysis
3. Regression Analysis


### Part 1 - Data Cleaning:
Libraries used: `Pandas`, `NumPy`, and `Matplotlib` 

**Goals**:
  1. Remove unnecessary data.
  2. Clean strings of unwanted characters.
  3. Recast columns to their appropriate types.
  4. Handle `Null` Values.

#### 1.1 Remove Unnecessary data that won't aid analysis:
 * Ex: removed the time at which someone responded to the survey.
 * Remove Non US Accountants.  
   * The survey was global, however given that my wife and I live in the US, it didn't make sense to have accountants from other  countries in the analysis.
     * 87% of the survey were accountants in the US, leaving 1201 accountants in total for the analysis.

#### 1.2 Cleaning Strings:
 * "Current Salary + Bonus"
   * Values like "73k" needed to be changed to "73000"
   * Replace commas and dollar signs with empty strings
 * "Line of Service"
   * Standardized entries by distilling 36 unique strings down to just 9:
     * Ex: Internal Audit, IT Audit -> Audit
     * Ex: Audit/Tax, Audit & tax, Tax and audit -> Tax and Audit
 * "Gender"
   * Standardized by distilling 2 unique strings down to 1:
     * Ex: Prefer not to say or Other -> Undisclosed

 #### 1.3 Converting columns to appropriate types 
 * "Current Salary + Bonus" to `float`
 * "Years of Experience" to `float`
 * "Average Hours Per Week" to `float`
 * "Years Public Before Exit" to `float`
 
 #### 1.4 Cleaning `Null` values
   * Gender: Replaced `Null` values with Undisclosed
   * Line of Service: Replaced `Null` values with Other
   * Current Industry: Replaced `Null` values with Other
   * Distribution of salaries was right-skewed, imputed `Null` values with the median.
   * Distribution of years of experience was right-skewed, imputed `Null` values with the median.
   * Cost of living: Imputed `Null` Values with the mode.

### Part 2 - Exploratory Data Analysis
Libraries used: `Pandas`, `NumPy`, `Matplotlib`, and `Seaborn`

| Qualitative data  | Quantitative data        |
| ----------------- | -----------------        |
| Exit status       | Years Experience         |
| Has CPA           | Current Salary + Bonus   |
| Current Industry  | Years Public Before Exit |
| Cost of Living    | Average Hours Per week   |
| Gender            |                          |
| Line of Service   |                          |

**Goals**:
  1. Understand the qualitative data.
  2. Understand different segments of accountants.
     * Segment by Gender
     * Segment by Line of Service
     * Segment by Industry
     * Segment by Has CPA
  4. Determine if those who have their CPA license earn more than those who don't.
  5. Determine if accountants who first started at a public accounting firm, then left for private industry, earn more than those who went directly into private industry.
  6. Understand accountants with extremely high salaries.
  7. 
