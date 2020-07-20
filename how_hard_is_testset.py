import numpy as np
import pandas as pd

csv_file = 'test.csv'
csv = np.array(pd.read_csv(csv_file, sep=',', header=None))


for x,y in csv:
    counter = 0
    for digit in x:
        if int(digit) != 0:
            counter += 1
    print(counter)
