import pandas as pd
#import data_processing
import matplotlib.pyplot as plt
#import anonymization 
import os
import numpy as np
import seaborn as sns

current_directory = os.getcwd()

def read_data(subd,k):
    subdirectory = subd
    #create file paths
    cleaned_data_path = os.path.join(current_directory, subdirectory, 'cleaned_'+subd+'.csv')
    mondrian_relaxed_path = os.path.join(current_directory, subdirectory, subd+'_mondrian_relaxed_'+str(k)+'.csv')
    mondrian_strict_path = os.path.join(current_directory, subdirectory, subd+'_mondrian_strict_'+str(k)+'.csv')
    u_mondrian_relaxed_path = os.path.join(current_directory, subdirectory, subd+'_u_mondrian_relaxed_'+str(k)+'.csv')
    u_mondrian_strict_path = os.path.join(current_directory, subdirectory, subd+'_u_mondrian_strict_'+str(k)+'.csv')
    #read datasets
    cleaned_data = pd.read_csv(cleaned_data_path, header=0, index_col=False, engine='python')
    mondrian_relaxed = pd.read_csv(mondrian_relaxed_path, header=0, index_col=False, engine='python')
    mondrian_strict = pd.read_csv(mondrian_strict_path, header=0, index_col=False, engine='python')
    u_mondrian_relaxed = pd.read_csv(u_mondrian_relaxed_path, header=0, index_col=False, engine='python')
    u_mondrian_strict = pd.read_csv(u_mondrian_strict_path, header=0, index_col=False, engine='python')

    return cleaned_data,mondrian_relaxed, mondrian_strict, u_mondrian_relaxed, u_mondrian_strict

# calculates for a given attribute its 'count' (in how many ranges it occours normalized by the lenght of the range)
# returns a dictionary with values of attribute and their corresponding count
# when called with sensitive it returns a dictioary with values of attribute and sensitive attribute as key and their corresponding count
def extract_counts(df, attribute, sensitive_attribute=None):
    counts = {}
    for _, row in df.iterrows():
        # reads attribute values
        attribute_value = row[attribute]
        if sensitive_attribute is not None:
            sensitive = row[sensitive_attribute]
        else:
            sensitive = None
        # convert range string to the start and end value as int 
        start, end = map(int, attribute_value.split('-'))
        for value in range(start, end + 1):
            if sensitive is not None:
                key = (value, sensitive)
            else:
                key = value
            # add value normalized by the size of the range to counts for specific key
            if key in counts:
                counts[key] += 1 / (end - start + 1)
            else:
                counts[key] = 1 / (end - start + 1)
    return counts


# split dictionary in dictionary of dictionaries, with a dictionary for each sensitive attribute  
def sensitive_counts(counts):
    distinct_dicts = {}
    for key, value in counts.items():
        sensitive_key = key[1]  # Get the second key from the tuple as key to find right dictionary 
        value_key = key[0] # new key for entries within sub dictionaries 
        if sensitive_key not in distinct_dicts:
            distinct_dicts[sensitive_key] = {}
        distinct_dicts[sensitive_key][value_key] = value
        sorted_dicts = dict(sorted(distinct_dicts.items())) #sort dictionaries according to the sensitive attribute
    return sorted_dicts


def overview_sensitive(dataset, attribute, sensitive_attribute, k):

    # read datasets
    original_data, mondrian_relaxed, mondrian_strict, u_mondrian_relaxed, u_mondrian_strict = read_data(dataset,k)
    
    datasets = [
    (original_data, 'original_data'),
    (mondrian_relaxed, 'mondrian_relaxed'),
    (mondrian_strict, 'mondrian_strict'),
    (u_mondrian_relaxed, 'u_mondrian_relaxed'),
    (u_mondrian_strict, 'u_mondrian_strict')
    ]  

    # create a figure including all plots
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9), sharex='col', sharey='row', gridspec_kw={'hspace': 0.3})
   
    # add labels above column
    for i, (_, title) in enumerate(datasets):
        fig.text(0.19 + i * 0.16, 0.90, title, ha='center', fontsize=12)
    colum_numbers = ['(i)','(ii)','(iii)','(iv)','(v)']
    for i, label in enumerate(colum_numbers):
        fig.text(0.19 + i * 0.16, 0.92, label, ha='center', fontsize=14)

    # add labels for rows
    row_labels = ['A', 'B', 'C']
    for i, label in enumerate(row_labels):
        fig.text(0.04, 0.87 - i * 0.28, label, va='center', ha='center', fontsize=16)

    # store handles and labels for legend
    handles = []
    labels = []

    # get attribute and unique sensitive attribute values
    sensitive_values = sorted(original_data[sensitive_attribute].unique())
    attribute_values = original_data[attribute]
    min_a = min(attribute_values)
    max_a = max(attribute_values)
    attribute_values = np.arange(min_a, max_a +1) # complete the attribute values that might be missing (to be continous)

    # values for histogram  
    bin_width = 3
    bins = np.arange(min_a, max_a + bin_width, bin_width) # creating array with bin edges

    # iterate over all datasets
    for i, (data, title) in enumerate(datasets):
        if i == 0: # original dataset

            # create references for bars
            bottom_ref = np.zeros(len(attribute_values))
            bar_positions = attribute_values

            # bar and line chart
            for sensitive in sensitive_values:
                # bar chart
                sensitive_data = original_data[original_data[sensitive_attribute] == sensitive]
                counts = sensitive_data[attribute].value_counts().reindex(attribute_values, fill_value=0) # fill in missing values
                bar = axes[1, i].bar(bar_positions, counts.values, bottom=bottom_ref, label = sensitive)

                # line chart
                axes[0, i].plot(counts.index, counts.values, label = sensitive)

               # handles and labels for legend
                handles.append(bar[0])
                labels.append(sensitive)
                bottom_ref += counts.values

            # histogram
            axes[2, i].hist(original_data[attribute],  bins=bins, color='skyblue', edgecolor='black', density=True)
        else: # anonymized datasets
            counts = extract_counts(data, attribute, sensitive_attribute)
            distinct_dicts = sensitive_counts(counts)

            # create references for bars
            bottom_ref = np.zeros(len(attribute_values))
            bar_positions = attribute_values
            
            # Line and bar chart
            for (sensitive, values) in distinct_dicts.items():
                sorted_values = list(values.items())       # the dictionray corresponding to the current value of the sensitive attribute
                x_values = [v[0] for v in sorted_values]     # the values of the attribute contained for this sensitve attribute
                # adding the missing values
                for value in attribute_values:
                    if value in x_values:
                        continue
                    else:
                        sorted_values.append((value, 0))
                sorted_values = sorted(sorted_values)
                x_values = [v[0] for v in sorted_values]    
                y_values = [v[1] for v in sorted_values]   
                #line chart
                axes[0, i].plot(x_values, y_values, label = sensitive)
                # bar chart
                axes[1, i].bar(bar_positions, y_values, bottom=bottom_ref, label = sensitive)
                bottom_ref += y_values

            # histogram
            # data provided as a dict (keys the values on x and values from dict the count for the density)
            counts = extract_counts(data, attribute)
            axes[2, i].hist(counts.keys(), weights=counts.values(), bins=bins, color='skyblue', edgecolor='black', density=True)

    # lables for axes
    for x in range(3):
        for y in range(5):
            axes[x, y].grid(axis='both', linestyle='--', linewidth='0.5', color='gray')
            axes[x, y].set_xlabel(attribute)
            axes[x, y].tick_params(axis='x', which='both', labelbottom=True)
            if y == 0:
                if x == 2:
                    axes[x, y].set_ylabel('Density')
                else:
                    axes[x, y].set_ylabel('Count')

    # create legend 
    fig.legend(handles, labels, loc='center right', ncol=1, fontsize=10, title=sensitive_attribute)

    # save plot
    plt.savefig(dataset+'_'+attribute+'_'+sensitive_attribute+'_'+str(k)+'.png')

    # show plot
    # plt.show()

overview_sensitive('adult','age','income',5)
#overview_sensitive('credit','AGE','default payment next month',5)
#overview_sensitive('diabetes', 'num_lab_procedures','admission_type_id', 5)

#overview_sensitive('diabetes', 'num_procedures','admission_type_id', 5)
 
def anonym_comparison(dataset, attribute, sensitive_attribute, k):

    # read datasets
    original_data, mondrian_relaxed, mondrian_strict, u_mondrian_relaxed, u_mondrian_strict = read_data(dataset,k)
    
    datasets = [
    (original_data, 'original_data'),
    (mondrian_relaxed, 'mondrian_relaxed'),
    (mondrian_strict, 'mondrian_strict'),
    (u_mondrian_relaxed, 'u_mondrian_relaxed'),
    (u_mondrian_strict, 'u_mondrian_strict')
    ]  

    # create a figure including all plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 9), sharex='col', sharey='row', gridspec_kw={'hspace': 0.3})
   
    # get attribute and unique sensitive attribute values
    sensitive_values = sorted(original_data[sensitive_attribute].unique())
    attribute_values = original_data[attribute]
    min_a = min(attribute_values)
    max_a = max(attribute_values)
    attribute_values = np.arange(min_a, max_a +1) # complete the attribute values that might be missing (in a continous)

    # lables for plots
    plot_labels = ["(ii)", "(iii)", "(iv)", "(v)"]

    # choose colors 
    x = len(sensitive_values)
    pastel = sns.color_palette("pastel")
    pastel = pastel[:x]
    bright = sns.color_palette("bright")
    bright = bright[:x]
    colors = pastel + bright
    for ax in axes.flat:
        ax.set_prop_cycle(color=colors)

    # collect handles and labels
    handles_labels = []

    for i, (data, title) in enumerate(datasets):
        if i==0: # original dataset
            for sensitive in sensitive_values:
                sensitive_data = original_data[original_data[sensitive_attribute] == sensitive]
                counts = sensitive_data[attribute].value_counts().reindex(attribute_values, fill_value=0) #fill in missing values
                # plot original in all plots 
                for x in range(2):
                    for y in range(2):
                        line, = axes[x, y].plot(counts.index, counts.values, label = sensitive)
                        # add handles and labels
                        if i == 0 and x == 0 and y == 0:
                            handles_labels.append((line, str(sensitive)+' original'))

        else: # anonymized datasets
            if i==1:
                x=0
                y=0
            elif i==2:
                x=1
                y=0
            elif i==3:
                x=0
                y=1
            else:
                x=1
                y=1
          
            counts_anonym = extract_counts(data, attribute, sensitive_attribute)    
            distinct_dicts = sensitive_counts(counts_anonym) 

            for (sensitive, values) in distinct_dicts.items():
                sorted_values = sorted(values.items())
                x_values = [v[0] for v in sorted_values]
                y_values = [v[1] for v in sorted_values]
                #line chart
                line,= axes[x, y].plot(x_values, y_values, label = sensitive)
                axes[x, y].grid(axis='both', linestyle='--', alpha=0.7)
                axes[x, y].set_xlabel(attribute)
                axes[x, y].tick_params(axis='x', which='both', labelbottom=True)
                axes[x, y].set_ylabel('Count')
                axes[x, y].set_title(title)

                # add labels to the top-right corner of each plot
                axes[x, y].text(0.95, 0.95, plot_labels[i-1], transform=axes[x, y].transAxes, fontsize=14, fontweight='bold', ha='right', va='top')

                # add handles and labels
                if i == 1 and x == 0 and y == 0:
                    handles_labels.append((line, str(sensitive)+' anonymized'))
    
    
    # extract handles and labels
    handles, labels = zip(*handles_labels)

    # create a legend for figure
    fig.legend(handles, labels, loc='upper center', ncol=len(sensitive_values), title=sensitive_attribute, fontsize='medium', title_fontsize='large')

    # save plot
    plt.savefig('closeup_'+dataset+'_'+attribute+'_'+sensitive_attribute+'_'+str(k)+'.png')

    # show the plot
    # plt.show()


anonym_comparison('adult','age','income',5)
#anonym_comparison('adult','hours-per-week','income',5)
#anonym_comparison('diabetes','number_diagnoses','readmitted',5)
#anonym_comparison('diabetes','num_medications','change',5)
 

def k_comparison(dataset,attribute, sensitive_attribute):
    k_values = [3,20,70]

    # create a figure including all plots
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 9), sharex='col', sharey='row', gridspec_kw={'hspace': 0.3})

    # labels for columns
    titles = ['mondrian_relaxed','mondrian_strict','u_mondrian_relaxed','u_mondrian_strict']
    for z, title in enumerate(titles):
        fig.text(0.2 + z * 0.205, 0.90, title, ha='center', fontsize=12)
    colum_numbers = ['(ii)','(iii)','(iv)','(v)']
    for i, label in enumerate(colum_numbers):
        fig.text(0.2 + i *  0.205, 0.92, label, ha='center', fontsize=14)

    # labels for rows
    for i, label in enumerate(k_values):
        fig.text(0.04, 0.87 - i * 0.28, 'k='+str(label), va='center', ha='center', fontsize=16)

    # legend handles and labels
    handles_labels = []

    for j, k in enumerate(k_values):
        # read datasets
        cleaned_data, mondrian_relaxed, mondrian_strict, u_mondrian_relaxed, u_mondrian_strict = read_data(dataset,k)
        
        datasets = [
        (cleaned_data, 'cleaned_data'),
        (mondrian_relaxed, 'mondrian_relaxed'),
        (mondrian_strict, 'mondrian_strict'),
        (u_mondrian_relaxed, 'u_mondrian_relaxed'),
        (u_mondrian_strict, 'u_mondrian_strict')
        ]  

        # get attribute and unique sensitive attribute values
        sensitive_values = sorted(cleaned_data[sensitive_attribute].unique())
        attribute_values = cleaned_data[attribute]
        min_a = min(attribute_values)
        max_a = max(attribute_values)
        attribute_values = np.arange(min_a, max_a +1) # complete the attribute values that might be missing (in a continous)

        # set colours
        x = len(sensitive_values)
        pastel = sns.color_palette("pastel")
        pastel = pastel[:x]
        bright = sns.color_palette("bright")
        bright = bright[:x]
        colors = pastel + bright
        for ax in axes.flat:
            ax.set_prop_cycle(color=colors)

        for i, (data,_) in enumerate(datasets):
            if i==0: # original data
                for sensitive in sensitive_values:
                    sensitive_data = cleaned_data[cleaned_data[sensitive_attribute] == sensitive]
                    counts = sensitive_data[attribute].value_counts().reindex(attribute_values, fill_value=0) #fill in missing values
                    # plot line in all plots
                    for y in range(4):
                        line, = axes[j, y].plot(counts.index, counts.values, label = sensitive)
                        # handles and labels for legend
                        if j == 0 and y == 0:
                            handles_labels.append((line, str(sensitive)+' original'))

            else: # anonymized data
                counts_anonym = extract_counts(data, attribute, sensitive_attribute)    
                distinct_dicts = sensitive_counts(counts_anonym)  

                for (sensitive, values) in distinct_dicts.items():
                    sorted_values = sorted(values.items())
                    x_values = [v[0] for v in sorted_values]
                    y_values = [v[1] for v in sorted_values]
                    # line chart
                    line, =axes[j, i-1].plot(x_values, y_values, label = sensitive)
                    axes[j, i-1].grid(axis='both', linestyle='--', alpha=0.7)
                    axes[j, i-1].set_xlabel(attribute)
                    axes[j, i-1].tick_params(axis='x', which='both', labelbottom=True)
                    axes[j, i-1].set_ylabel('Count')
                    # handles and labels for legend
                    if i == 1 and j == 0:
                        handles_labels.append((line, str(sensitive)+' anonymized'))

    # extract the handles and labels
    handles, labels = zip(*handles_labels)

    # create legend for figure
    fig.legend(handles, labels, loc='center right', ncol=1, title=sensitive_attribute, fontsize='medium', title_fontsize='large')

    # save plot
    plt.savefig('k_comparison'+dataset+'_'+attribute+'_'+sensitive_attribute+'.png')

    # show plot
    # plt.show()

#k_comparison('credit','AGE','default payment next month')
#k_comparison('diabetes','num_procedures','change')

k_comparison('adult','age','income')
#k_comparison('adult','hours-per-week','income')
#k_comparison('diabetes','num_medications','change')
#k_comparison('diabetes','num_lab_procedures','change')
#k_comparison('diabetes','number_diagnoses','change')


def dataset_comparison(k):
    data = ['adult','credit','diabetes']

    # create a figure including all plots
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 9), sharex='col', sharey='row', gridspec_kw={'hspace': 0.3})

   
    # column labels
    titles = ['mondrian_relaxed','mondrian_strict','u_mondrian_relaxed','u_mondrian_strict']
    for z, t in enumerate(titles):
        fig.text(0.2 + z * 0.205, 0.90, t, ha='center', fontsize=12)
    colum_numbers = ['(ii)','(iii)','(iv)','(v)']
    for i, label in enumerate(colum_numbers):
        fig.text(0.2 + i *  0.205, 0.92, label, ha='center', fontsize=14)

    # rows labels
    row_labels = ['A', 'B', 'C']
    for i, label in enumerate(row_labels):
        fig.text(0.04, 0.87 - i * 0.28, label, va='center', ha='center', fontsize=16)

    for j,dataset in enumerate(data):
        # read datasets
        original_data, mondrian_relaxed, mondrian_strict, u_mondrian_relaxed, u_mondrian_strict = read_data(dataset,k)
        
        datasets = [
        (original_data, 'original_data'),
        (mondrian_relaxed, 'mondrian_relaxed'),
        (mondrian_strict, 'mondrian_strict'),
        (u_mondrian_relaxed, 'u_mondrian_relaxed'),
        (u_mondrian_strict, 'u_mondrian_strict')
        ]  

        if dataset=='adult':
            attribute='age'
            sensitive_attribute='income'
        elif dataset=='credit':
            attribute='AGE'
            sensitive_attribute='default payment next month'
        else:
            attribute='num_medications'
            sensitive_attribute='readmitted'

        # get attribute and unique sensitive attribute values
        sensitive_values = sorted(original_data[sensitive_attribute].unique())
        attribute_values = original_data[attribute]
        min_a = min(attribute_values)
        max_a = max(attribute_values)
        attribute_values = np.arange(min_a, max_a +1) # complete the attribute values that might be missing (in a continous)

        
        # set colors
        if j==0:
            pastel = sns.color_palette("pastel")
            pastel = pastel[:2]
            bright = sns.color_palette("bright")
            bright = bright[:2]
            colors = pastel + bright
        elif j==1:
            pastel = sns.color_palette("pastel")
            pastel = pastel[2:4]
            bright = sns.color_palette("bright")
            bright = bright[2:4]
            colors = pastel + bright
        else:
            pastel = sns.color_palette("pastel")
            pastel1 = pastel[6:8] 
            pastel2 = [pastel[9]]
            bright = sns.color_palette("bright")
            bright1 = bright[6:8]
            bright2 = [bright[9]]
            colors = pastel1 + pastel2 + bright1 + bright2
        for ax in axes.flat:
            ax.set_prop_cycle(color=colors)

        # handles and labels for legend
        handles_labels = []

        for i, (data, _) in enumerate(datasets):
            if i==0: # original data
                for sensitive in sensitive_values:
                    sensitive_data = original_data[original_data[sensitive_attribute] == sensitive]
                    counts = sensitive_data[attribute].value_counts().reindex(attribute_values, fill_value=0) #fill in missing values
                    # plot original in all plots
                    for y in range(4):
                        line, = axes[j, y].plot(counts.index, counts.values, label = sensitive)
                        if y==0:
                            handles_labels.append((line, str(sensitive)+' original'))
            else:
                counts_anonym = extract_counts(data, attribute, sensitive_attribute)    
                distinct_dicts = sensitive_counts(counts_anonym)  

                for (sensitive, values) in distinct_dicts.items():
                    sorted_values = sorted(values.items())
                    x_values = [v[0] for v in sorted_values]
                    y_values = [v[1] for v in sorted_values]
                    # line chart
                    line, = axes[j, i-1].plot(x_values, y_values, label = sensitive)
                    axes[j, i-1].grid(axis='both', linestyle='--', alpha=0.7)
                    axes[j, i-1].set_xlabel(attribute)
                    axes[j, i-1].tick_params(axis='x', which='both', labelbottom=True)
                    axes[j, i-1].set_ylabel('Count')
                    if i == 1:
                        handles_labels.append((line, str(sensitive)+' anonymized'))
    
        # extract handles and labels
        handles, labels = zip(*handles_labels)

        # titles for each legend
        legend_titles = ['income', 'default payment', 'readmitted']

        # legend for each row
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.89, 0.89 - 0.28 * j), fontsize='small', title=legend_titles[j])

    # save plot
    plt.savefig('dataset_compariason_med'+str(k)+'.png')

    # show the plot
    # plt.show()

dataset_comparison(10)
