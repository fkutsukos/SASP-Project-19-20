import pandas as pd
import numpy as np
import os


def save_to_csv(echoes, csv_name='output.csv', number_of_microphones=5):
    column_names = ["score"]
    for i in range(number_of_microphones):
        column_names.append("distance_" + str(i))

    df = pd.DataFrame(columns=column_names)
    for index, echo, in enumerate(echoes):
        df.loc[index] = [echo[0], *echo[1]]
    outputs_path = '../outputs'
    df.to_csv(os.path.join(outputs_path, csv_name), index=False)


def load_from_csv(csv_name):
    outputs_path = '../outputs'
    loaded_df = pd.read_csv(os.path.join(outputs_path, csv_name), index_col=None)
    records = loaded_df.to_records(index=False)
    loaded_echoes = []

    for record in records:
        loaded_echoes.append((record[0], np.array([record[i + 1] for i in range(len(loaded_df.columns) - 1)])))

    return loaded_echoes
