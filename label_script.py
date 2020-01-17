import string
import re
import numpy as np
import pandas as pd
import torch
import unicodedata
import glob

# Formats:
# 0 = first last
# 1 = first middle last
# 2 = last, first
# 3 = last, first middle
# 4 = first middle_initial. last
# 5 = last, first middle_initial.
FORMAT_REGEX = ["^(\w[A-Za-z'-]+\s\w[A-Za-z'-]+)$", "^(\w[A-Za-z'-]+\s\w[A-Za-z'-]+\s\w[A-Za-z'-]+)$",
                "^(\w[a-zA-Z-']+,\s\w[a-zA-Z-']+)$", "^(\w[a-zA-Z-']+,\s\w[a-zA-Z-']+\s\w[a-zA-Z-']+)$",
                "^(\w[A-Za-z'-]+\s[A-Z]+.\s\w[A-Za-z'-]+)$", "^(\w[A-Za-z'-]+,\s\w[A-Za-z'-]+\s[A-Z]+\.)$"]
PREFIX_TITLES = ['mr', 'miss', 'mrs', 'dr', 'professor']
POST_NOM_TITLES = ['jr', 'sr', 'i', 'ii', 'iii', 'iv', 'v', 'phd', 'md', 'ph.d']

def label_name_data(df: pd.DataFrame, col_hdr: str = "name", delimiter: str = " ", name_hdr: str = "name", fn_hdr: str = "first",
                    mn_hdr: str = "middle", ln_hdr: str = "last", format_hdr: str = "format"):
    """
    Takes in dataframe of name and labels the data's first, middle and last name by splitting each name by delimiter
    then taking the index assigned to each subname in parameters(fn_idx, mn_idx, ln_idx) and assigning it to row
    Args:
    df: dataframe to be labelled
    col_hdr: The column in df that contains names
    delimiter: Character to split name into subnames
    name_hdr: Header of name of return DF
    fn_hdr: Header of first name of return DF
    mn_hdr: Header of middle name of return DF
    ln_hdr: Header of last name of return DF
    format_hdr: Header of format of return DF
    """
    names_list = list()
    pattern_0 = re.compile(FORMAT_REGEX[0])
    pattern_1 = re.compile(FORMAT_REGEX[1])
    pattern_2 = re.compile(FORMAT_REGEX[2])
    pattern_3 = re.compile(FORMAT_REGEX[3])
    pattern_4 = re.compile(FORMAT_REGEX[4])
    pattern_5 = re.compile(FORMAT_REGEX[5])

    for _, row in df.iterrows():
        name = row[col_hdr]
        first, middle, last = np.nan, np.nan, np.nan
        list_element = ()

        lower_name = name.lower()

        is_title = any(substring in lower_name for substring in POST_NOM_TITLES) or any(
            substring in lower_name for substring in PREFIX_TITLES)

        # Formats:
        # 0 = first last
        # 1 = first middle last
        # 2 = last, first
        # 3 = last, first middle
        # 4 = first middle_initial. last
        # 5 = last, first middle_initial.
        if is_title:
            continue
        elif pattern_0.match(name):
            split_name = name.split()
            list_element = (name, split_name[0], '', split_name[1], 0)
        elif pattern_1.match(name):
            split_name = name.split()
            list_element = (name, split_name[0], split_name[1], split_name[2], 1)
        elif pattern_2.match(name):
            split_name = name.split()
            list_element = (name, split_name[1], '', split_name[0][:-1], 2)
        elif pattern_3.match(name):
            split_name = name.split()
            list_element = (name, split_name[1], split_name[2], split_name[0][:-1], 3)
        elif pattern_4.match(name):
            split_name = name.split()
            list_element = (name, split_name[0], split_name[1][0], split_name[2], 4)
        elif pattern_5.match(name):
            split_name = name.split()
            list_element = (name, split_name[1], split_name[2][0], split_name[0][:-1], 5)
        else:
            continue

        print(list_element)
        names_list.append(list_element)

    return pd.DataFrame(names_list, columns=[name_hdr, fn_hdr, mn_hdr, ln_hdr, format_hdr])

def create_labeled_csv(path : str = "data", save_file : str = "labaelled_data"):
    """
    all CSV files in data should only have one column named "name"
    """
    all_files = glob.glob(path + "/*.csv")

    all_data = []

    for i in all_files:
        df = pd.read_csv(i)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
            print(df)
        all_data.append(df)

    frame = pd.concat(all_data, axis=0, ignore_index=True)

    frame = label_name_data(frame)
    frame.drop_duplicates(subset="name", keep = False, inplace= True)
    frame.to_csv(f'data/{save_file}.csv')

create_labeled_csv()
df = pd.read_csv('data/labelled_data.csv')
print(df["format"].value_counts())
