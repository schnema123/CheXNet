import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv("../nihcc/data_test.csv")

print(csv[csv['Patient Age'] == 1])
print(csv['Finding Labels'].value_counts())

csv.hist(bins=50, figsize=(20, 15))
plt.show()