"""
This script reads a CSV file, splits the data into three equal parts,
and then saves each part to a new CSV file.

The script starts by importing pandas and defining the path to the CSV file.
It then reads the data using pandas
read_csv function. The data is split into three parts, each of which is approximately
one third of the original dataset.
Each of these parts is then saved to a new CSV file.

Constants:
FILE_PATH (str): The path to the CSV file that will be read and split.

Functions:
None

Usage:
This script is intended to be run as a standalone file and does not take any command line arguments.

Example:
$ python split_data.py

This script doesn't return anything but creates three new CSV files
in the same directory as the original CSV file,
each containing approximately one third of the original dataset.
"""

import pandas as pd

FILE_PATH = "../data/test.csv"

# read the data
data = pd.read_csv(FILE_PATH)

# split the data into 3 parts
thrid = int(len(data) / 3)
data_1 = data.iloc[:thrid]
data_2 = data.iloc[thrid : 2 * thrid]
data_3 = data.iloc[2 * thrid :]

# save the data
data_1.to_csv(
    "../data/test_1.csv",
    index=False,
)
data_2.to_csv(
    "../data/test_2.csv",
    index=False,
)
data_3.to_csv(
    "../data/test_3.csv",
    index=False,
)
