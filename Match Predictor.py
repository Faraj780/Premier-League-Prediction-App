import pandas as pd
matches = pd.read_csv("match_data.csv",index_col=0)
print(matches.shape)
print(matches["team"].value_counts())