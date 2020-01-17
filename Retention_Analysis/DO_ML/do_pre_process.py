#!/usr/bin/env python3

import DO_ML
from DO_ML.do_input import *
from DO_ML.do_visual import *
from DO_ML.do_common_libraries import *
import os
import csvkit


def column_distribution(df, column_name):
    return df[column_name].value_counts()


def column_size(df):
    return df.shape[1]


def convert_to_csv(file, sheet=0):
    import agate
    # file_extension = pathlib.Path(file).suffix
    file_path, file_extension = os.path.splitext(file)
    csvfile = file_path + ".csv"
    try:
        if file_extension == ".xls":
            import agateexcel
            table = agate.Table.from_xls(file, sheet=sheet)
            table.to_csv(csvfile)
        elif file_extension == ".xlsx":
            import agateexcel
            table = agate.Table.from_xlsx(file, sheet=sheet)
            table.to_csv(csvfile)
        elif file_extension == ".json":
            table = agate.Table.from_json(file, key=file.key)
            table.to_csv(csvfile)
        else:
            return

    except 'FileNotFoundError':
        return "File not found, please try again"
    except 'PermissionError':
        return "File in use"


def drop_columns(df, column_list=[]):
    return df.drop(columns=column_list)


def fill_na_with_mean(df):
    return df.fillna(df.mean(), inplace=True)


def find_missing(df):
    missing = df[df.isnull().any(axis=1)]
    if missing.empty:
        return "None"
    return missing


def find_row(df, column_name, row_value):
    return df.loc[df[column_name] == row_value]


def get_chi_square(column1, column2):  # chi square for independence
    from scipy.stats import chi2_contingency
    df = pd.crosstab(column1, column2)
    chi2, p, dof, expected = chi2_contingency(df.values)
    return chi2, p, dof, expected


def get_columnlist_from_dtypes(df, type_list):
    return list(df.select_dtypes(type_list).columns)


def get_columnlist_from_excluded_dtypes(df, type_list):
    return list(df.select_dtypes(exclude=type_list).columns)


def get_pearsonr_coefficient_pvalue(column1, column2):
    from scipy.stats import pearsonr
    return pearsonr(column1, column2)

def get_spearmanr_coefficient_pvalue(column1, column2):
    from scipy.stats import spearmanr
    return spearmanr(column1, column2)


def group_by_column(df, column_list=[]):
    grouped = df.groupby(column_list)
    group_mean = df.groupby(column_list).mean()
    group_sum = df.groupby(column_list).sum()
    return grouped, group_mean, group_sum


def hot_encode(df, column_list):
    categorical = column_list
    return pd.get_dummies(df, columns=categorical, drop_first=True)


def make_dataframe(array=[], column_list=None):

    df = pd.DataFrame(array, columns=column_list)
    return df


def map_dataframe_header(df_without, df_with):
    df_without.columns = df_with.columns
    return df_without


def normalize_x(train, test, scaler="standard"):
    if scaler == "standard":
        norm_scaler = StandardScaler()
        return norm_scaler.fit_transform(train), norm_scaler.transform(test)
    elif scaler == "minmax":
        norm_scaler = MinMaxScaler(feature_range=(0,1))
        return norm_scaler.fit_transform(train), norm_scaler.transform(test)


def round_num(num, precision=0):
    if type(num) == int:
        return num
    elif type(num) == float:
        if precision > 0:
            return float("{0:.{1}f}".format(num, precision))
        else:
            if num < 0:
                return int(num - 0.5)
            return int(num + 0.5)
    else:
        if precision == 0:
            def func1(x):
                if x<0:
                    res = int(x-0.5)
                    return res
                res = int(x+0.5)
                return res
            result = list(map(lambda x: func1(x), num))
            return result
        else:
            res = [float("{0:.{1}f}".format(i, precision)) for i in num]
            return res


def row_size(df):
    return df.shape[0]


def rebalance_data(x_df, y_df):
    from imblearn.over_sampling import SMOTE
    x_array, y_array = SMOTE().fit_sample(x_df, y_df.values.ravel())
    return x_array, y_array


def split_dataset(df, target_column_name, size=0.8):
    """ size: size of the training set"""
    X = df.drop([target_column_name], axis=1).values
    y = df[target_column_name].values

    return train_test_split(X, y, test_size=1-size, random_state=7)


def summarize_df(df):
    rows_ = "Rows: , {0}".format(df.shape[0])
    columns_ = "\nColumns: , {0}".format(df.shape[1])
    features_ = "\nFeatures: \n{0}".format(df.columns.tolist())
    missing_ = "\nMissing Values: \n{0}".format(df.isnull().sum().values.sum())
    unique_ = "\nUnique Values: \n{0}".format(df.nunique())
    print(rows_, columns_, features_, missing_, unique_)
    return f"DataFrame has {missing_} total missing values across all columns"


def summary_stats(df):
    return df.describe()


def main():
    pass

if __name__ == '__main__': main()