#!/usr/bin/env python
# coding: utf-8

# In[8]:


def report_nulls(df, verbose=False):
    """
    Show a fast report of the DF.
    If Verbose=True, column type will be printed to output as well
    """
    rows = df.shape[0]
    columns = df.shape[1]
    null_cols = 0
    list_of_nulls_cols = []
    for col in list(df.columns):
        null_values_rows = df[col].isnull().sum()
        null_rows_pcn = round(((null_values_rows)/rows)*100, 2)
        col_type = df[col].dtype
        if null_values_rows > 0:
            print('The column {} has {} null values.  It is {}% of total rows.'.format(col, null_values_rows, null_rows_pcn))
            if verbose==True:
                print('The column {} is of type {}.\n'.format(col, col_type))
            null_cols += 1
            list_of_nulls_cols.append(col)
            print()
    null_cols_pcn = round((null_cols/columns)*100, 2)
    print('The Data Frame has {} columns with null values.  It is {}% of total columns.'.format(null_cols, null_cols_pcn))
    return list_of_nulls_cols


# In[ ]:




