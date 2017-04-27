# PROGRAM NAME: Solution.py
# PROGRAM PURPOSE: Demonstrating SPAM/HAM classification with the Naive Bayes algorithm
# PROGRAMMER: Dillon Pietsch (Copied from Udacity's github page: https://github.com/udacity/machine-learning/blob/master/projects/practice_projects/naive_bayes_tutorial/Naive_Bayes_tutorial.ipynb)
# DATE WRITTEN: 4-27-17

import pandas as pd

# Dataset downloaded from https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])

# Output printing the first 5 rows
print(df.head())

# Map label column with categories (SPAM/HAM) to 1/0.
df['label'] = df.label.map({'ham': 0, 'spam': 1})
print()
print(df.shape)  # Get a feel for how big the dataset is
print()
print(df.head())  # See our new label column
