#!/usr/bin/env python3

import DO_ML
from DO_ML.do_common_libraries import *
from DO_ML import do_pre_process
# import pathlib
import os
import xlrd


class InputSeries(pd.Series):

    @property
    def _constructor(self):
        return InputSeries

    @property
    def _constructor_expanddim(self):
        return InputDataFrame


class InputDataFrame(pd.DataFrame):

    _internal_names = pd.DataFrame._internal_names + ["file", "db_table"]   # will not be passed to manipulated results
    _internal_names_set = set(_internal_names)

    _metadata = ["size"]

    @property
    def _constructor(self):
        return InputDataFrame

    @property
    def _constructor_sliced(self):
        return InputSeries

    def get_csvfile(self, file, sheet=0, **input_kwargs):
        file_path, file_extension = os.path.splitext(file)
        do_pre_process.convert_to_csv(file, sheet)
        csvfile = file_path + ".csv"
        return csvfile

    def set_db_uri_string(self):
        from dbconfig import db_uri_string
        uri_string = db_uri_string["uri"]
        return uri_string

    def get_input(self, file, psql_db=False, db_table="", input_kwargs={}):
        try:
            csvfile = self.get_csvfile(file)
            if not psql_db:
                input_df = pd.read_csv(csvfile, encoding="ISO-8859-1", **input_kwargs)
            else:
                uri_string = self.set_db_uri_string()
                input_df = pd.read_sql_table(db_table, uri_string)
            return input_df

        except 'FileNotFoundError':
            return "csv file does not exist. Please convert file to csv"
        except 'PermissionError':
            return "File in use"
        except:
            return "File format may not be supported at this time. Please first convert to csv format (use: convert_to_csv(file, sheet)) and try again"


def main():
    pass

if __name__ == '__main__': main()