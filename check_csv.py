import pandas as pd

df = pd.read_csv('C:/Users/kplom/Documents/thesis/thesis_outputs/master_index.csv')
print(f'Total rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print()
print(df.groupby(['dataset','split','label_name']).size().to_string())
print()
print(f'Missing seg: {(df.has_seg == "no").sum()}')
print(f'Missing modalities: {(df.has_missing == "yes").sum()}')