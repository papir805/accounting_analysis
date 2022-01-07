def get_group_stats(df, by_label, target_label, label):
    label_df = df[df[by_label]==label]
    print('Descriptive stats for {} - {}:'.format(target_label, label))
    print('_'*60)
    print(label_df[target_label].describe().to_string())
    print()
    print()



## Original concept for the distribution_plotter function in cell below
# plt.figure(figsize=(10,5))
# for gender in us_accounting['Gender'].unique():
#     by_gender = us_accounting[us_accounting['Gender']==gender]
#     plt.hist(
#         by_gender['Current Salary + Bonus'], 
#         bins=np.arange(0, 300_001, 20_000), 
#         alpha=0.5, 
#         label=gender
#     )
#     print('Descriptive stats for Current Salary + Bonus, Gender:', gender)
#     print(by_gender['Current Salary + Bonus'].describe().to_string())
#     print()
#     print()
# plt.legend()
# plt.xlabel('Current Salary + Bonus')
# plt.ylabel('Count')
# plt.title('Distribution of Salaries + Bonus by Gender');





# Original code for distribution_plotter function
def distribution_plotter(df, by_label, target_label, bins, verbose=False, normal=False):
    """
    df: DataFrame object to get data from
    by_label: String of Column name that you want to group by
    target_label: Column name of numerical data you want to use for plotting
    bins: Range object that specifies the distributions bin intervals
    verbose: Boolean that turns additional Descriptive Stats on
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,5))
    for label in df[by_label].unique():
        label_df = df[df[by_label]==label]
        plt.hist(
            label_df[target_label], 
            bins=bins, 
            alpha=0.5, 
            label=label,
            edgecolor='black',
            density=normal
    )
        if verbose==True:
            get_group_stats(df, by_label, target_label, label)
    plt.legend()
    plt.xlabel(target_label)
    if normal==True:
        plt.ylabel('Density')
        plt.title('Proportion of {} by {}'.format(target_label, by_label))
    else:
        plt.ylabel('Count')
        plt.title('Distribution of {} by {}'.format(target_label, by_label))
    plt.tight_layout();
    
    
    
    
# # Original Concept for additional code in the distribution_plotter_fancy function to plot side by side graphs on subplots
# fig, axes = plt.subplots(figsize=(10,5), nrows=1, ncols=len(us_accounting['Gender'].unique()), sharey=True)
# us_accounting[us_accounting['Gender'] == 'Male']['Current Salary + Bonus'].plot(kind='hist', ax = axes[0], bins=np.arange(0, 300001, 20000), color='blue')
# us_accounting[us_accounting['Gender'] == 'Female']['Current Salary + Bonus'].plot(kind='hist', ax = axes[1], bins=np.arange(0, 300001, 20000), color='orange')
# us_accounting[us_accounting['Gender'] == 'Undisclosed']['Current Salary + Bonus'].plot(kind='hist', ax = axes[2], bins=np.arange(0, 300001, 20000), color='green')
# axes[0].set_title('Male Accountants')
# axes[0].set_xlabel('Current Salary + Bonus')
# axes[0].grid()
# axes[1].set_title('Female Accountants')
# axes[1].set_xlabel('Current Salary + Bonus')
# axes[1].grid()
# axes[2].set_title('Undisclosed Gender Accountants')
# axes[2].set_xlabel('Current Salary + Bonus')
# axes[2].grid()
# plt.tight_layout();



# Modified code to produce overlaid histograms, or side by side histograms, still need to fix the verbose part to work with the each for loop. 
def distribution_plotter_fancy(df, by_label, target_label, bins, verbose=False, overlaid=True, normal=False):
    """
    df: DataFrame object to get data from
    by_label: String of Column name that you want to group by
    target_label: Column name of numerical data you want to use for plotting
    bins: Range object that specifies the distributions bin intervals
    verbose: Boolean that turns additional Descriptive Stats on
    Overlaid: Boolean that when true plots all distributions on one axes overlaid with each other,
    when False, will plot the distributions on their own axes
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    if overlaid == True:
        for label in df[by_label].unique():
            label_df = df[df[by_label]==label]
            plt.hist(
                label_df[target_label], 
                bins=bins, 
                alpha=0.5, 
                label=label,
                edgecolor='black',
                density=normal
            )
            if verbose==True:
                get_group_stats(df, by_label, target_label, label)
        plt.legend()
        plt.xlabel(target_label)
        if normal==True:
            plt.ylabel('Density')
            plt.title('Proportion of {} by {}'.format(target_label, by_label))
        else:
            plt.ylabel('Count')
            plt.title('Distribution of {} by {}'.format(target_label, by_label))
        plt.tight_layout();
        
        
    elif overlaid==False:
        i = 0
        num_cols = df[by_label].nunique()
        colors = ['blue', 'orange', 'green']
        #print(num_cols)
        fig, axes = plt.subplots(
            figsize=(10,5), 
            nrows=len([target_label]), # Cast to list so it has length equal to number of labels, not length of string 
            ncols=num_cols, 
            sharey=True
        )
        #print(axes.shape)
        for label, color in zip(df[by_label].unique(), colors):
            label_df = df[df[by_label]==label]
            #print(i)
            label_df[target_label].plot(kind='hist', ax=axes[i], bins=bins, color=color, edgecolor='black')
            axes[i].set_title('{} Gender Accountants'.format(label))
            axes[i].set_xlabel('{}'.format(target_label))
            axes[i].grid()
            plt.show()
            i += 1
            #print(i)

            if verbose==True:
                get_group_stats(df, by_label, target_label, label)

                
                
                
# ## This is the original concept code for the group_scatter_plotter function below and the group_scatter_plotter_fancy function when all_groups=False
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(los_groups.get_group('Advisory')['Years Experience'], los_groups.get_group('Advisory')['Current Salary + Bonus']);


def group_scatter_plotter(x_label, y_label, grouped_object, group_label, figsize=(10,10)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(grouped_object.get_group(group_label)[x_label], grouped_object.get_group(group_label)[y_label])
    plt.xlabel(x_label, fontsize=figsize[0]*2)
    plt.ylabel(y_label, fontsize=figsize[1]*2)
    plt.title('{} and {} for {}'.format(x_label, y_label, group_label))
    plt.grid();
    
    
    
    
# ## This is the original concept code for the group_scatter_plotter_fancy function below, when all_groups = True
# los_groups = us_accounting.groupby('Line of Service')
# fig, ax = plt.subplots(figsize=(10,10))
# for name, group in los_groups:
#     ax.scatter(group['Years Experience'], group['Current Salary + Bonus'], marker='o', label=name, alpha =0.4)
# plt.legend();



### Note, group_label is needed for individual scatter plot, however not needed when all_groups = True and all groups go onto same scatter plot.  Need to clean this up and figure out how to implement code correctly.
# Could do something like this: group_label = list(los_groups.groups.keys())[0] where we have group_label automatically assume to plot the 1st element of the grouped object by casting the dictionary keys to a list and accessing it's first element.
# For now I've put a try/except block in to handle the KeyError that results when the group_label arg is not passed.
def group_scatter_plotter_fancy(x_label, y_label, grouped_object, group_label='', figsize=(10,10), all_groups=False):
    """
    Creates a scatter plot from a pandas.groupby object.  If all_groups = True, 
    then all of the groups are plotted on the same scatter plot.  
    If all_groups = False, the x,y data will be plotted from only one group, dictated by group_label"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    if all_groups == False:
        try:
            ax.scatter(grouped_object.get_group(group_label)[x_label], grouped_object.get_group(group_label)[y_label])
            plt.title('{} and {} for {}'.format(x_label, y_label, group_label))
        except KeyError:
            print('group_label is needed, please provide a group_label when all_groups=False to prevent this error')
    elif all_groups == True:
        for name, group in grouped_object:
            ax.scatter(group[x_label], group[y_label], marker='o', label = name, alpha=0.4)
            plt.legend()
            plt.title('{} and {} for all US Accountants'.format(x_label, y_label))
    plt.grid()    
    plt.xlabel(x_label, fontsize=figsize[0]*2)
    plt.ylabel(y_label, fontsize=figsize[1]*2);
