# split /Users/husseinyounes/University/big_data/San-Francisco-Crime-Classification-bigdata/data/train.csv into smaller files
# divide it into 3 files
# call them train_1.csv, train_2.csv, train_3.csv
# each file should have 1/3 of the data

import pandas as pd

file_path = '/Users/husseinyounes/University/big_data/San-Francisco-Crime-Classification-bigdata/data/test.csv'

# read the data
data = pd.read_csv(file_path)

# split the data into 3 parts
thrid = int(len(data)/3)
data_1 = data.iloc[:thrid]
data_2 = data.iloc[thrid:2*thrid]
data_3 = data.iloc[2*thrid:]

# save the data
data_1.to_csv('/Users/husseinyounes/University/big_data/San-Francisco-Crime-Classification-bigdata/data/test_1.csv', index=False)
data_2.to_csv('/Users/husseinyounes/University/big_data/San-Francisco-Crime-Classification-bigdata/data/test_2.csv', index=False)
data_3.to_csv('/Users/husseinyounes/University/big_data/San-Francisco-Crime-Classification-bigdata/data/test_3.csv', index=False)
