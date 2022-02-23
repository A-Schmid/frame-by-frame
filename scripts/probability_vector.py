import config
import os
from VideoHandler import VideoHandler
import pandas as pd

def generate_probability_vector(category):
    files = [f for f in os.listdir(f'{config.RESULT_PATH}/{category}')]
    handlers = []
    for f in files:
        name = f.split('.')[0]
        handlers.append(VideoHandler(category, name))

    v = VideoHandler.get_probability_vector_for_category(category)
    return v

out_path = f'{config.DATA_PATH}/probability_vector.csv'

categories = os.listdir(config.RESULT_PATH)

vector_list = []

counter = 1

for category in categories:
    print(f'{counter}/{len(categories)}')
    counter += 1
    v = generate_probability_vector(category)
    v['category'] = category
    vector_list.append(v)

df = pd.DataFrame(vector_list)
df = df.sort_values(by=['category'])

# category goes first
cols = df.columns.tolist()
cols = [cols[-1]] + cols[:-1]
df = df.reindex(columns=cols)

df.to_csv(out_path, index=False)
