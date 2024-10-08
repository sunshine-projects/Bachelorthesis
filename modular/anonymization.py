'''
The anonymization module receives a df without missing values and information about the category of the attributes
it anonymizes the data through strict or relaxe Mondrian or u-Mondrian to achieve k-anonymity
it returns a df or saves the dataset to csv file being k-anonymious and having attributes saved as ranges
'''

import pandas as pd
import numpy as np
import data_processing
from sklearn.neighbors import LocalOutlierFactor


df=pd.DataFrame()
names_of_attributes = []
categorical_attributes = []
numerical_attributes = []
qid = []
sensitive_attributes = []
k = 0

'''receiving information for the dataframe (from the data processing module)'''
def set_df_with_info(df_in,datainfo_in):
    global df,names_of_attributes,categorical_attributes,numerical_attributes,qid,sensitive_attributes
    df = df_in
    names_of_attributes = datainfo_in['names of attributes']
    categorical_attributes = datainfo_in['categorical']
    numerical_attributes = datainfo_in['numerical']
    qid = datainfo_in['qid']
    sensitive_attributes = datainfo_in['sensitive']

def set_k(parameter_k):
    global k
    k = parameter_k



# help functions for mondrian and u-mondrian
def calc_range (df, partition, attribute): 
    at_range = df.loc[partition, attribute].max() - df.loc[partition, attribute].min()
    return (at_range)

# find dimension with max range
def max_range (df, partition, attributes): 
    ranges = []
    for a in attributes:
        ranges.append((calc_range(df, partition, a),a))      
    return (max(ranges)[1]) # returns attribute with biggest range 

def calc_frequencies(df, partition, attribute): 
    frequency = {}
    for a in df.loc[partition, attribute]:
        if a in frequency:
            frequency[a] +=1
        else:
            frequency[a] =1
    sorted_freq = dict(sorted(frequency.items())) 
    return sorted_freq # return sorted frequency dictionary

def calc_median(freq_dict):
    # convert frequency dictionary to list with the number of entries per key equivalent to the frequency to find median      
    freq_list = []
    for value, frequency in freq_dict.items(): 
        freq_list.extend([value]*frequency)
    l = len(freq_list)
    middle = l//2
    if l%2 == 1:
        median = freq_list[middle]
    else:
        median = (freq_list[middle]+freq_list[middle-1])/2
    return median   # returns value of median not index

# check for k-anonymity (partition needs to have at least cardinality k)
def k_anonymous (partition): 
    if len(partition) < k: 
        return False
    else:
        return True

# split strict
def split_s(df, partition, attribute, median): # initial dataframe, partitionindexes, the attribute (qid) to split over and the median for that attribute
    dfsplit = df.loc[partition, attribute] 
    lpartition = []
    rpartition = []
    for index, value in dfsplit.items():
        if value <= median:
            lpartition.append(index)
        else:
            rpartition.append(index)
    return(lpartition, rpartition) # returns indexes of lpartition and rpartition 


# split relaxed 
def split_r(df,partition,attribute,median):
    dfsplit = df.loc[partition, attribute] 
    lpartition = []
    rpartition = []
    count = 0 # create a counter to split every sencond element that is on the median to lpartition and the other one to rpartition
    for index, value in dfsplit.items():
        if value == median and count == 0:
            lpartition.append(index)
            count = 1
        elif value == median and count == 1:
            rpartition.append(index)
            count = 0
        elif value < median:
            lpartition.append(index)
        else:
            rpartition.append(index)
    return(lpartition, rpartition) # returns indexes of lpartition and rpartition 

def calc_partitions(df,partition,split):
    p_working = [partition] # list of lists initialized with the list of the indeces of the partition that should be partitioned
    p_done = [] # list where partitions that cannot be split more are collected
    while p_working:
        dfw = p_working.pop(0) 
        attribute = max_range(df, dfw, qid) 
        median = calc_median(calc_frequencies(df, dfw , attribute))
        if split == 'strict':
            lpartition,rpartition = split_s(df, dfw, attribute, median)
        else:
            lpartition,rpartition = split_r(df, dfw, attribute, median)
        if k_anonymous (lpartition) and k_anonymous (rpartition):   
            p_working.append(lpartition) 
            p_working.append(rpartition)
        else:                                                                     
            p_done.append(dfw) 
    return p_done # list of lists with the indices of the partitions is returned


def generalize(df, partition):
    dfa = df.copy()                     
    for attribute in qid:
        # need to change type from int to string to be able to store range
        dfa[attribute] = dfa[attribute].astype(str)        
    for dfw in partition:     # set of indexes which is to be anonymized 
        for attribute in qid:                   
            max = df.loc[dfw, attribute].max()
            min = df.loc[dfw, attribute].min()
            dfa.loc[dfw, attribute] = str(min) + '-' + str(max)      
    return dfa                                                  



# help function for u-mondiran
def lof_and_knn(df, partition):
    if len(partition) <= k:                    
        return (partition, []) 
    else:
        # finding reference point
        subset_df = df.loc[partition][qid] 
        if k >= 20:
            n = 20
        else:
            n = k
        lof = LocalOutlierFactor(n_neighbors=n)   # number of neighbours considered
        lof.fit(subset_df)
        lof_scores = lof.negative_outlier_factor_      
        max_index = np.argmax(lof_scores)      # the max value (closest to 0) is point with highest density 
        ref = subset_df.iloc[max_index]      # iloc because the index of subset_df is not continous
        # finding k-1 nearest neighbours
        distances = np.linalg.norm(subset_df - ref, axis=1) 
        sorted_indexes = np.argsort(distances)          
        normal_indexes = sorted_indexes[:k]           
        outlier_indexes = sorted_indexes[k:]         
        normal_df=subset_df.iloc[normal_indexes]        # indexlists are with positional index but need to return the indeces of normal and outliers in the original df
        outlier_df=subset_df.iloc[outlier_indexes]
    return (normal_df.index, outlier_df.index)          # return touple of lists with normal and outlier indices

def mondrian(df,split): # specify if strict or relaxed partitioning should be used
    partition_indices = calc_partitions(df,df.index,split)
    anonym_df = generalize(df, partition_indices)
    return anonym_df

def u_mondrian(df,split):
    # initialization
    partition = calc_partitions (df,df.index,split)
    normal_set = []
    outlier_set = []
    partition_all = [df.index]
    for p in partition:
        normal,outlier = lof_and_knn(df, p)
        normal_set.append(normal)           
        outlier_set.extend(outlier)
    # loop
    while len(outlier_set) >= k:
        partition_outlier = calc_partitions (df,outlier_set,split)
        outlier_set = []  
        partition_all = partition_outlier + normal_set       # create a set of partitions with the normal set and the partition of the current outlierset
        # preapare normal and outliers for next iteration
        for p in partition_outlier:
            normal,outlier = lof_and_knn(df, p)
            normal_set.append(normal)       
            outlier_set.extend(outlier)
    dfa = generalize(df,partition_all)
    return dfa

# saving anonymized dataset to not have to recalculate it every time
def export_df(df,name):
    df.to_csv(name+'.csv')






#################################### preset dataset anonymization #################################
'''
# creating and saving the anonymized datasets as a csv to visualize them in the visualization module 
# without having to recalculate them every time, as the anonymization is not highly efficent

values = [70,60,50,40,30,20,10,5,3]
partitioning = ['strict','relaxed']
algorithm = ['mondrian','u_mondrian']

# creating anonymizations for the adult dataset
dfx,datainfo = data_processing.adult()
set_df_with_info(dfx,datainfo)
for value in values:
    for parti in partitioning:
        for algo in algorithm:
            set_k(value)
            # can be add print statement to follow the anonymization prozess, as this might take long
            # print('adult',value,parti,algo)
            if algo == 'mondrian':
                result = mondrian(df,parti)
            else:
                result = u_mondrian(df,parti)
            export_df(result,'adult_'+ algo +'_'+parti+'_'+str(value))

# creating anonymizations for the credit dataset
dfx,datainfo = data_processing.credit()
set_df_with_info(dfx,datainfo)
for value in values:
    for parti in partitioning:
        for algo in algorithm:
            set_k(value)
            # can be add print statement to follow the anonymization prozess, as this might take long
            # print('credit',value,parti,algo)
            if algo == 'mondrian':
                result = mondrian(df,parti)
            else:
                result = u_mondrian(df,parti)
            export_df(result,'credit_'+ algo +'_'+parti+'_'+str(value))

# anonymization of the diabetes dataset
dfx,datainfo = data_processing.diabetes()
set_df_with_info(dfx,datainfo)
for value in values:
    for parti in partitioning:
        for algo in algorithm:
            set_k(value)
            # can be add print statement to follow the anonymization prozess, as this might take long
            # print('diabetes',value,parti,algo)
            if algo == 'mondrian':
                result = mondrian(df,parti)
            else:
                result = u_mondrian(df,parti)
            export_df(result,'diabetes_'+ algo +'_'+parti+'_'+str(value))
'''
##############################################################################################################




#################################### metric functions #############################################
# these functions are not used in the implementation
#  they are provided however to complete the transparensy of the study and to be able to add them if desired 

# discernability metric and average equivalence class size can be measured before generalizing
def dm (p_done):
    sum = 0
    for partition in p_done:
        cardinality = len(partition)
        sum = sum + cardinality ** 2
    return sum

def aecs (df,p_done):
    result = len(df) / (len(p_done) * k)
    return result


# the generalized certainty penalty must be measured after generalizing
def gcp(df, dfa, p_done):
    dim = len(qid)
    t = len(df)
    sum_total = 0
    # calculating ncp for each equivalence class 
    for p_working in p_done:  
        sum_ncp = 0
        for attribute in qid:
            # calculate for each qid      
            range_df = calc_range(df, df.index, attribute)
            #look at first element in partition as they all have same ranges on quasiidentifiers
            working_index = p_working[0]                      
            values = str(dfa[attribute][working_index]).split('-')
            min_val = int(values[0])
            max_val = int(values[1])
            # calc range on attribute over this partiton           
            range_p = max_val - min_val                  
            result = range_p / range_df
            sum_ncp = sum_ncp + result
        size = len(p_working)
        sum_total = sum_total + sum_ncp * size  
    result = sum_total / (dim * t)
    return result
#########################################################################################






dfx,datainfo = data_processing.new()
set_df_with_info(dfx,datainfo)
values = [60,70]
partitioning = ['strict','relaxed']
algorithm = ['u-mondrian']
for value in values:
    for parti in partitioning:
        for algo in algorithm:
            set_k(value)
            # can be add print statement to follow the anonymization prozess, as this might take long
            # print('adult',value,parti,algo)
            if algo == 'mondrian':
                result = mondrian(df,parti)
            else:
                result = u_mondrian(df,parti)
            print('done')
            export_df(result,'imports_'+ algo +'_'+parti+'_'+str(value))