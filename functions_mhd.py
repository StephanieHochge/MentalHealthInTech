"""this module contains self-defined functions that are used in several other scripts to analyze the 'mental health in tech dataset'"""

# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools


# self-defined functions used in several scripts
def create_uniques_df(data):
    """create a data frame containing the variable names, their unique values and their counts for each column in data
    
    :param data:  the dataset for which the data frame is to be created ('pandas.core.frame.DataFrame')
    :return:  a data frame containing the unique values and their counts for each column in data ('pandas.core.frame.DataFrame')
    """
    
    uniques = [data[feature].unique().tolist() for feature in data]
    df_uniques = pd.DataFrame(data.columns.tolist(), columns=['variable'])
    df_uniques['unique_values'] = uniques
    df_uniques['uniques_count'] = [len(val) for val in uniques]
    return df_uniques


def explore_feature(feature, fig_size=(6.4, 4.8), original=True):  
    """create a table for the feature displaying absolute and relative value counts and visualize the value counts in a bar plot
    
    :param feature:  the feature for which the value count is to be displayed and visualized ('pandas.core.series.Series')
    :param fig_size:  width and height of the bar plot in inches ('(float, float)')
    :param original:  if the feature was included in the original dataset, the question asked is displayed as the plot's title
    """
    
    # load the dictionary containing the variable names the corresponding questions asked in the survey
    df_dict_question_labels = pd.read_csv('dict_question_labels.csv')
    var_names = df_dict_question_labels['var_name'].tolist()
    questions = df_dict_question_labels['question'].tolist()
    dict_questions_labels = dict(zip(var_names, questions))
    
    feature_counts = feature.value_counts(dropna=False, normalize=False)
    df_counts = pd.DataFrame(feature_counts).rename(columns={feature.name:'count'})
    df_counts['percent'] = feature.value_counts(dropna=False, normalize=True)
    print('value counts of', feature.name, ':')
    display(df_counts)
    if original:
        feature_counts.plot(kind='barh', figsize=fig_size, title=dict_questions_labels[feature.name])
    else:
        feature_counts.plot(kind='barh', figsize=fig_size)
    plt.show()
    

def identify_data_type(uniques):
    """identify the data type (nominal or ordinal) of a set of response options based on certain criteria, e.g., variables with a "not applicable" (n/a) option are automatically regarded as
    nominal. 
    
    :param uniques:  the set of response options ('set')
    :return:  the data type of the set of response options ('str')
    """
    
    uniques_str = [str(val) for val in uniques]
    ordinal_vals = ['Maybe', 'Maybe/Not sure', 'More than 1000', 'Neither easy nor difficult', 'I don\'t know']
    na_vals = ['Not applicable to me', 'Not eligible for coverage / N/A', 'Not applicable to me (I do not have a mental illness)', 
               'n/a', 'N/A because no employer provided coverage']
    if any(val in uniques_str for val in na_vals):
        return 'nominal'
    elif len(uniques_str) == 2 or len(uniques_str) > 10:
        return 'nominal'
    elif any(val in uniques_str for val in ordinal_vals):
        return 'ordinal'
    elif set(uniques_str) == {'Sometimes', 'Never', 'Always'}:
        return 'ordinal'
    elif set(uniques_str) == {'0.0', '1.0', 'nan'}:
        return 'nominal'
    elif 'other' in uniques_str:
        return 'nominal'
    elif any(val in uniques_str for val in ['2', '3', '4', '-1', '-2']):
        return 'ordinal'
    else:
        return 'nominal'

    
def count_missings(data, response_cat='missing'):
    """create a data frame containing the absolute and relative counts of missing values in data
    
    :param data:  the data in which to count the missing values ('pandas.core.frame.DataFrame')
    :param response_cat:  the type of response category reflecting missing values in the dataset ('str'). This can also be "not applicable" responses that have been recoded as missing values.
    :return:  a data frame containing the absolute and relative counts of missing values in data ('pandas.core.frame.DataFrame')
    """
    
    df_missings_per_column = data.isnull().sum().to_frame().reset_index()
    df_missings_per_column.rename(columns={'index':'variable', 0:f'{response_cat}_count'}, inplace=True)
    df_missings_per_column[f'{response_cat}_percent'] = df_missings_per_column[f'{response_cat}_count']/len(data)
    return df_missings_per_column


def define_as_missings(data, response_cat):
    """fill the missing values in a dataset with "missing" and define a certain response category as missing values for easy statistical analysis
    
    :param data:  the data in which to define responses of a certain type as missing values ('pandas.core.frame.DataFrame')
    :param response_cat:  the type of response category that is to be defined as missing values ('str'). 
    :return:  a data frame where a certain response category is defined as missing values for easy statistical analysis ('pandas.core.frame.DataFrame')
    """
    
    response = 'I don\'t know' if response_cat == 'dk' else ['Not applicable to me', 'Not eligible for coverage / N/A',
                                                         'Not applicable to me (I do not have a mental illness)', 'n/a', 
                                                         'N/A because no employer provided coverage']
    df_responses = data.copy()
    df_responses.fillna("missing", inplace=True)
    return df_responses.replace(response, np.nan)


def count_responses(data, response_cat):
    """create a data frame containing the absolute and relative counts of "don't know" or "not applicable" response categories in data
    
    :param data:  the data in which to count the "don't know" or "not applicable" response categories ('pandas.core.frame.DataFrame')
    :param response_cat:  the type of response category that is to be counted ('str'). 
    :return:  a data frame containing the absolute and relative counts of the "don't know" or "not applicable" response categories in data ('pandas.core.frame.DataFrame')
    """
    
    df_responses = define_as_missings(data, response_cat)
    return count_missings(df_responses, response_cat)
    
    
def create_data_dict(data):
    """ create a data dictionary for the data containing information about the unique values, unique value counts, the data type, absolute and 
    relative counts of missing values, "don't know" options and "not applicable" options.
    
    :param data:  the data for which the data dictionary is to be created ('pandas.core.frame.DataFrame')
    :return:  the data dictionary ('pandas.core.frame.DataFrame')
    """
    
    # identify the data type of variables in data
    df_uniques = create_uniques_df(data)
    d_types = [identify_data_type(uniques) for uniques in df_uniques['unique_values'].tolist()]
    df_uniques['data_types'] = d_types
    
    # manually change data types of ratio variables to ratio
    index_ratio = df_uniques.index[df_uniques['variable'].str.contains('count_|_len|suicide_rate', na=False, regex=True)]
    df_uniques.loc[index_ratio, 'data_types'] = 'ratio'
    df_uniques.loc[df_uniques['variable'] == 'age', 'data_types'] = 'ratio'
    
    # add information about the relative and absolute counts of missing values, "don't know" responses and "not applicable" responses
    df_missings = count_missings(data)
    data_dict = pd.merge(df_uniques, df_missings, on='variable')
    
    df_dk = count_responses(data, 'dk')
    data_dict = pd.merge(data_dict, df_dk, on='variable')
    
    df_not_app = count_responses(data, 'n/a')
    data_dict = pd.merge(data_dict, df_not_app, on='variable')
    
    return data_dict


def missing_statistics(data, response_cat="missing"):
    """ display statistics for the occurrence of certain response types (e.g., missing values, "don't know" values or "not applicable" values).
    
    :param data:  the data for which the statistics are to be displayed ('pandas.core.frame.DataFrame')
    :param response_cat:  the response types for which statistics are to be displayed ('str')
    """
    
    df_missings_per_column = count_missings(data, response_cat)
    print(f"Number of variables with {response_cat} values:")
    print(df_missings_per_column[df_missings_per_column[f'{response_cat}_count']>0]['variable'].count())
    print(f"\nDescriptive statistics of percent {response_cat} values per column:")
    print(df_missings_per_column[f'{response_cat}_percent'].describe())
    
    # calculate the percentage of the values of interest for each dataset
    print(f"\nDescriptive statistics of {response_cat} values per row:")
    df_missings_rows = data.isnull().sum(axis=1)
    display(df_missings_rows.describe())

        
def code_as_inconsistent(indexes, reason, data):
    """ code certain records as inconsistent, specifying the reason
    
    :param indexes:  the indexes of the records to be coded as inconsistent ('pandas.core.indexes')
    :param reason: the reason why the records are inconsistent ('str')
    :param data:  the data source ('pandas.core.frame.DataFrame')
    """
    
    for idx in indexes:
        data.at[idx, 'times_inconsistent'] += 1
        data.at[idx, 'reason_inconsistent'] += reason
    
    
def display_mhd_data(index, data, focus='all'): 
    """display mental health disorder data for specific respondents
    
    :param index: the index of the respondents for whom the mental health disorder data is to be displayed ('pandas.core.indexes')
    :param data: the data source ('pandas.core.frame.DataFrame')
    :param focus: determines whether all data are to be displayed or e.g. current diagnoses ('str')
    :return: the filtered mental health disorder data ('pandas.core.frame.DataFrame')
    """
    
    if focus == 'all':
        columns = [col for col in data.columns if '(1)' in col or '(2)' in col or '(3)' in col]
    else:
        focus_dict = {'curr_diag':'(1)', 'poss_diag': '(2)', 'prof_diag': '(3)'}
        columns = [col for col in data.columns if f'{focus_dict[focus]}' in col]
    return data.loc[index, ['count_diagnosed_illnesses', 'count_possible_illnesses', 'count_prof_diagnoses', 'self_mental_illness_present', 'self_mental_illness_past', 
                            'diagnosed_by_professional'] + columns]


def show_pivot_tables_and_bars(col_rows, col_cols, data):
    """create and show a pivot table as well as grouped bar diagram for two columns of a dataset
    
    :param col_rows: the name of the column to constitute the rows of the pivot table ('str')
    :param col_cols: the name of the column to constitute the columns of the pivot table ('str')  
    :param data: the data source of the columns ('pandas.core.frame.DataFrame')
    """
    
    # create the pivot table containing absolute values
    df_no_missings = data.fillna('missing').copy() # because the pivot_table function would otherwise drop missing values
    table = pd.pivot_table(df_no_missings, index=col_rows, columns=col_cols, aggfunc='size', dropna=False)
        
    # create the pivot table containing relative values
    table_percent = table.div(table.sum(1), 0) # to divide each cell by its row sum
    
    # show the table and plot bar diagrams
    print('absolute values:')
    display(table)
    table.plot(kind='bar')
    plt.show()
    print('-'*42)
    print('relative values:')
    display(table_percent)
    table_percent.plot(kind='bar')
    plt.show()


def split_by_line(feature):
    """split the values of feature into a list with several elements using the separator "|" 
    
    :param feature: the feature whose values are to be split ('pandas.core.series.Series')
    :return: a list of lists containg the individual elements formerly separated by the separator "|" ('list')
    """
    
    return [str(val).split("|") for val in feature]


def n_per_option(data, col):
    """display a frequency table for a column where individuals could select multiple options and visualize value counts 
    
    :param data: the data which contains the column of interest ('pandas.core.frame.DataFrame')
    :param col: the name of the column of interest ('str')
    """
    
    options_all = split_by_line(data[col])
    options_all_flat = pd.Series(list(itertools.chain.from_iterable(options_all)), name=col)
    explore_feature(options_all_flat, fig_size=(16,12), original=False)