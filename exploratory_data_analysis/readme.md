# Part 2 - Exploratory Data Analysis (EDA)
Important libraries used: `Pandas`, `NumPy`, `Matplotlib`, and `Seaborn`

| Qualitative data  | Quantitative data        |
| ----------------- | -----------------        |
| Exit status       | Years Experience         |
| Has CPA           | Current Salary + Bonus   |
| Current Industry  | Years Public Before Exit |
| Cost of Living    | Average Hours Per week   |
| Gender            |                          |
| Line of Service   |                          |

## How to use this repository:
[Click here](https://github.com/papir805/accounting_analysis/blob/main/exploratory_data_analysis/accounting_eda.ipynb) or [here](https://nbviewer.org/github/papir805/accounting_analysis/blob/main/exploratory_data_analysis/accounting_eda.ipynb) to view the Jupyter Notebook that has all the code used, as well as the outputs and visualizations I produced.

## Goals:
  1. Understand the quantitative data and what they tell us about the typical US accountant.
  2. Understand different segments of accountants using the qualitative data.
     * Segment by Gender
     * Segment by Line of Service
     * Segment by Industry
     * Segment by Has CPA
  3. Use Has CPA to determine if those who have their CPA license earn more than those who don't.
  4. Use Exit Status to determine if accountants who first started at a public accounting firm, then left for private industry, earn more than those who went directly into private industry.  
  5. Understand accountants with extremely high salaries (outliers).

## Key Findings:
1. The typical US accountant: 
    * Has mean earnings between $83,000-$89,000 annually or median earnings of roughly $72,000-$73,000.
    * Has between 3-4 years of experience.
    * If the accountant worked in Public accounting and has exited, they have spent between 3-5 years before doing so.
    * Works an average of 45-46 hours per week.
2. Based on segmentation:
    * An overwhelming majority of respondents to the survey are male (roughly 3.5:1) and there is significant evidence to indicate a pay gap between Male and Female accountants exists across all of the most commong Lines of Service and many of the most common industries.
3. Do those who have their CPA earn more than those who don't?
    * **YES!**  At least the data from our survey indicates this is the case.  While the proportion of accountants who have their CPA is roughly equivalent to the proportion that don't (roughly 1:1), those who have their CPA license realize a very tangible increase in their yearly salary.  (Mean: ~$24,000/yr and Median: ~$22,00/yr more)
4. Do those who started in a public accounting firm, then left for private industry, earn more than those who went directly into private industry?
    *  Our survey data indicates that those who started in at a public firm first see significant increases in salary.  (Mean: ~$25,000/yr and Median: ~$25,000/yr more).
5. Accountants considered to be outliers by their salaries tend to have much more work experience, but show similar hours worked per week as compared to their peers.  However, **the pay gap among genders seems to be very pronounced here with male accountants earning as much as several hundred of thousands of dollars more as compared to female accountants, despite having equal, or in some cases less experience.**

### Conclusion: 
My wife had done a lot of research on her own as to the best plan to navigate here way through the accounting field and the data from this survey seem to support her findings.  If one's primary goal is to maximize their earning potential in the accounting field, one should:
 * **Get their CPA license.**
 * **Start their career by spending several years at a public firm, then transition to private industry.**
