# Part 1 - Data Cleaning:
Libraries used: `Pandas`, `NumPy`, and `Matplotlib` 

**Goals**:
  1. Remove unnecessary data.
  2. Clean strings of unwanted characters.
  3. Recast columns to their appropriate types.
  4. Handle `Null` Values.

## 1.1 Remove Unnecessary data that won't aid analysis:
 * Ex: removed the time at which someone responded to the survey.
 * Remove Non US Accountants.  
   * The survey was global, however given that my wife and I live in the US, it didn't make sense to have accountants from other  countries in the analysis.
     * 87% of the survey were accountants in the US, leaving 1201 accountants in total for the analysis.

## 1.2 Cleaning Strings:
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

 ## 1.3 Converting columns to appropriate types 
 * "Current Salary + Bonus" to `float`
 * "Years of Experience" to `float`
 * "Average Hours Per Week" to `float`
 * "Years Public Before Exit" to `float`
 
 ## 1.4 Cleaning `Null` values
   * Gender: Replaced `Null` values with Undisclosed
   * Line of Service: Replaced `Null` values with Other
   * Current Industry: Replaced `Null` values with Other
   * Distribution of salaries was right-skewed, imputed `Null` values with the median.
   * Distribution of years of experience was right-skewed, imputed `Null` values with the median.
   * Cost of living: Imputed `Null` Values with the mode.
