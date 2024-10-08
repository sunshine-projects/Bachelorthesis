'''
The data_processing module reads a dataset and prepares it for the anonymization process by removing
missing values and classifying the types of attributes 
it can save the cleaned dataset or pass it with or without attribute information
It offers 3 prepared datasets and additional functionalities that can be used to study a new dataset
'''

import pandas as pd
import os

'''global variables'''
file_path = ""
names_attributes = []
categorical_attributes = []
numerical_attributes = []
qid_attributes = []
sensitive_attributes = []
missing_symbol = ""
current_directory = os.getcwd()


'''getting user input for dataset'''
def set_file_path():
    file_path = input("Please enter the file path: ")
    return file_path.strip()

def set_names_attributes():
    string = input("Please enter the names of the attributes if not specified in the header separated by commas, if they are specified, type the number of the row which contains them (typically 0 or 1): ")
    # split the input string by commas and strip each attribute to remove spaces
    return [attribute.strip() for attribute in string.split(',')]

def set_categorical_attributes():
    string = input("Please enter the names of categorical attributes separated by commas: ")
    return [attribute.strip() for attribute in string.split(',')]

def set_numerical_attributes():
    string = input("Please enter the names of numerical attributes separated by commas: ")
    return [attribute.strip() for attribute in string.split(',')]

def set_qid_attributes():
    string = input("Please enter the names of  attributes used as quasi-identifiers separated by commas: ")
    return [attribute.strip() for attribute in string.split(',')]

def set_sensitive_attributes():
    string = input("Please enter the names of attributes that are sensitive separated by commas: ")
    return [attribute.strip() for attribute in string.split(',')]

def set_missing_symbol():
    missing_symbol = input("Enter the symbol representing missing values: ")
    return missing_symbol.strip()


'''setting variables, reading and cleaning df'''
def read_df():
    global file_path, names_attributes
    file_path = set_file_path()
    names_attributes = set_names_attributes()
    # check if dataset includes the names of the attributes and in which row 
    if len(names_attributes) == 1 and names_attributes[0].isdigit():
        # read df using the provided number as the row index for the names of the attributes
        row = int(names_attributes[0])
        df = pd.read_csv(file_path, header=row, index_col=False, engine='python')
    else:
        # read df using the list of strings as names for the attributes
        df = pd.read_csv(file_path, header = None, names = names_attributes, index_col = False, engine = 'python')
    names_attributes = df.columns.tolist()
    return df

# set variables according to user-input
def set_attributes():
    global categorical_attributes,numerical_attributes,qid_attributes,sensitive_attributes
    categorical_attributes = set_categorical_attributes()
    numerical_attributes = set_numerical_attributes()
    qid_attributes = set_qid_attributes()
    sensitive_attributes = set_sensitive_attributes()

# create ability to return the attribute info
def get_info():
    attribute_info = {
        'names of attributes': names_attributes,
        'categorical': categorical_attributes,
        'numerical': numerical_attributes,
        'qid': qid_attributes,
        'sensitive': sensitive_attributes
    }
    return attribute_info

def clean_df(df):
    global missing_symbol
    missing_symbol = set_missing_symbol()
    if missing_symbol.lower() == 'nan':
        df.dropna(inplace=True)
    else:
        for attribute in df.columns:
            # identify rows with missing_symbol values and remove whitespaces
            missing_rows = df[df[attribute].astype(str).str.strip() == missing_symbol]
            df.drop(missing_rows.index, inplace=True)
    return df

# not a necessary function can be used to determine how many missing values the attributes in a given dataset have
def count_missing_values(df):
    missing_values = {}
    global missing_symbol
    missing_symbol = set_missing_symbol()
    for attribute in df.columns:
        # Count the number of missing values in attribute
        missing_count = df[attribute].eq(missing_symbol).sum() 
        missing_values[attribute] = missing_count
    return missing_values

'''options for returning cleaned datasets'''
# returning the cleaned data as a df
def cleaned_data():
    df=read_df()
    clean=clean_df(df)
    return clean

# returning the cleaned data as a df including information on categories of attributes
def cleaned_data_with_info():
    df=read_df()
    set_attributes()
    info=get_info()
    clean=clean_df(df)
    return clean,info

# saving the cleaned data to a csv file
def save_cleaned_data():
    df=read_df()
    clean=clean_df(df)
    clean.to_csv('cleaned_data.csv')


'''preset datasets: adult, credit and diabetes
getting prepared, cleaned and then saved and returned as cleaned versions and the corresponding info of the dataset'''
def adult():
    global file_path, names_attributes, categorical_attributes,numerical_attributes,qid_attributes,sensitive_attributes, missing_symbol
    subdirectory = 'adult'
    file_path = os.path.join(current_directory, subdirectory, 'adult.data.txt')
    names_attributes=['age','workclass', 'final-weight', 'education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
    df = pd.read_csv(file_path, header = None, names = names_attributes, index_col = False, engine = 'python')
    names_attributes = df.columns.tolist()
    categorical_attributes = ['workclass','education','marital-status','occupation','relationship','sex','native-country','race','income']
    numerical_attributes = ['education-num','age','final-weight','capital-gain','capital-loss','hours-per-week']
    qid_attributes = ['age','final-weight','capital-gain','capital-loss','hours-per-week']
    sensitive_attributes = ['income']
    missing_symbol = '?'
    for attribute in df.columns:
        missing_rows = df[df[attribute].astype(str).str.strip() == missing_symbol]
        df.drop(missing_rows.index, inplace = True)
    info=get_info()
    df.to_csv('cleaned_adult.csv')
    return (df,info)

def diabetes():
    global file_path, names_attributes, categorical_attributes,numerical_attributes,qid_attributes,sensitive_attributes, missing_symbol
    subdirectory = 'diabetes'
    file_path = os.path.join(current_directory, subdirectory, 'diabetic_data.csv')
    df = pd.read_csv(file_path, header = 0, index_col = False, engine = 'python')
    names_attributes = df.columns.tolist()
    categorical_attributes = ['race','gender','age','medical_specialty','max_glu_serum','A1Cresult','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton,insulin','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','change','diabetesMed','readmitted']
    numerical_attributes = ['encounter_id','patient_nbr','weight','admission_type_id','discharge_disposition_id','admission_source_id','time_in_hospital','payer_code','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','diag_1','diag_2','diag_3','number_diagnoses']
    qid_attributes = ['num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses']
    sensitive_attributes = ['change','readmitted','admission_type_id']
    missing_symbol = '?'
    # had to remove some collums which had too many missing values form the dataset entiery
    columns_to_remove = ['weight', 'payer_code', 'medical_specialty']
    df.drop(columns = columns_to_remove, inplace = True)
    for attribute in df.columns:
        missing_rows = df[df[attribute].astype(str).str.strip() == missing_symbol]
        df.drop(missing_rows.index, inplace = True)
    info = get_info()
    df.to_csv('cleaned_diabetes.csv')
    return (df,info)

def credit():
    global file_path, names_attributes, categorical_attributes,numerical_attributes,qid_attributes,sensitive_attributes, missing_symbol
    subdirectory = 'credit'
    file_path = os.path.join(current_directory, subdirectory, 'default_of_credit_card_clients.csv')
    df = pd.read_csv(file_path, header = 1, index_col = False, engine = 'python')
    names_attributes = df.columns.tolist()
    categorical_attributes = []
    numerical_attributes = ['ID','LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default payment next month']
    qid_attributes = ['SEX','EDUCATION','MARRIAGE','AGE']
    sensitive_attributes = ['default payment next month']
    info = get_info()
    df.to_csv('cleaned_credit.csv')
    return (df,info)

def new():
    global file_path, names_attributes, categorical_attributes,numerical_attributes,qid_attributes,sensitive_attributes, missing_symbol
    file_path = os.path.join(current_directory, 'imports-85.data')
    names_attributes=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    df = pd.read_csv(file_path, header = None, names = names_attributes, index_col = False, engine = 'python')
    names_attributes = df.columns.tolist()
    categorical_attributes = []
    numerical_attributes = []
    qid_attributes = ['b', 'n', 'q', 'v', 'w', 'x', 'y', 'z']
    sensitive_attributes = ['d']
    missing_symbol = '?'
    for attribute in df.columns:
        missing_rows = df[df[attribute].astype(str).str.strip() == missing_symbol]
        df.drop(missing_rows.index, inplace = True)
    df[qid_attributes] = df[qid_attributes].astype(int)
    info=get_info()
    df.to_csv('cleaned_imports.csv')
    return (df,info)

