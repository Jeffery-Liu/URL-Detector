import random
import sys

import pandas as pd

data = pd.DataFrame(pd.read_csv("data_validation.csv", ",", error_bad_lines=False))
# data_validation = data.sample(n=20464)
print(data)
# print("Validation Data Size: ", len(data_validation))
# Converting it into an array
random.shuffle(data)
print(data)
pd.DataFrame(data).to_csv("data_t.csv", index=False)
# pd.DataFrame(data).to_csv("data_validation.csv", index=False)

# pd.DataFrame(data).to_csv("data_index.csv")


