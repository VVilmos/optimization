import pandas as pd

x = [1, 2, 3, 4, 5]

df = pd.DataFrame(x)

df.to_csv('test.csv', index=False, header=False)