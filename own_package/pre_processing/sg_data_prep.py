import pandas as pd
import numpy as np
import os
import xlrd
import openpyxl
from own_package.others import print_array_to_excel
from pathlib import Path


def stack_columns(directory, target_columns, target_rows):
    # Get list of file name within the directory that contains all the demand excel file.
    file_name_store = []
    directory = directory
    for idx, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        file_name_store.append(directory + '/' + filename)
    print('Loading the following models from {}. Total excel files = {}'.format(directory, len(file_name_store)))

    # Check if target_columns and/ or target_rows are numbers or list
    if not all(isinstance(i, list) for i in target_rows):
        temp = [item for item in [target_rows] for i in range(len(file_name_store))]
        target_rows = temp

    if not all(isinstance(i, list) for i in target_columns):
        temp = [item for item in [target_columns] for i in range(len(file_name_store))]
        target_columns = temp

    # list containing all the columns stacked into one list
    stacked = []

    for idx, item in enumerate(zip(file_name_store, target_columns, target_rows)):
        file_name, target_column, target_row = item
        try:
            df = pd.read_excel(file_name)
        except xlrd.biffh.XLRDError:
            df = pd.read_csv(file_name)
            pass
        for column in target_column:
            stacked.extend(df.iloc[target_row, column].values.tolist())
    print('Len of stacked column ={}'.format(len(stacked)))
    return stacked


def sg_data():
    excel_path = r'C:/Users/User/Desktop/Python/CN5111 - Copy/excel'
    demand_path = excel_path + '/sg_demand'
    price_path = excel_path + '/sg_price'
    stacked_demand = stack_columns(demand_path,
                                   target_columns=[2, 5, 8, 11, 14, 17, 20],
                                   target_rows=list(range(3, 51)))
    stacked_price = stack_columns(price_path,
                                  target_columns=[3],
                                  target_rows=[list(range(0, 672)),
                                               list(range(0, 1344)),
                                               list(range(0, 1344)),
                                               list(range(0, 1344)),
                                               list(range(0, 1344)),
                                               list(range(0, 1344)),
                                               list(range(0, 1344))])

    excel_name = excel_path + '/results.xlsx'
    wb = openpyxl.Workbook()
    wb.save(excel_name)
    sheetname = wb.sheetnames[-1]
    ws = wb[sheetname]

    # Writing other subset split, instance per run, and bounds
    print_array_to_excel(['price'], (1,1), ws, axis=0)
    print_array_to_excel(['demand'], (1, 2), ws, axis=0)
    start_row = 2
    start_col = 1
    print_array_to_excel(np.array(stacked_price), (start_row, start_col), ws, axis=0)
    print_array_to_excel(np.array(stacked_demand), (start_row, start_col + 1), ws, axis=0)
    wb.save(excel_name)
    wb.close()
