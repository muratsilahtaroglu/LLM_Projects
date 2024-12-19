import pandas as pd
import random
path = "files/question_answering_normal_and_reversed_data.csv"
df = pd.read_csv(path)
seed_value = random.randint(1, 1000) 
