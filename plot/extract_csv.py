import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


"""
Adapted from https://stackoverflow.com/questions/42355122/can-i-export-a-tensorflow-summary-to-csv/52095336
"""


def tabulate_events(dirs):
    # summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for directory in os.listdir(dpath)]
    summary_iterators = [EventAccumulator(directory).Reload() for directory in dirs]

    out = defaultdict(list)

    steps = [e.step for e in summary_iterators[0].Scalars('social_welfare')]
    print('len of steps', len(steps))
    for tag in ['social_welfare']:

        scalars = []
        for acc in summary_iterators:
            try:
                scalars.append(acc.Scalars(tag))
            except KeyError:
                continue
        scalars_new = []

        for scalar in scalars:
            while len(scalar) % len(steps) != 0:
                scalar.append(scalar[-1])
            if len(scalar) > len(steps):
                freq = len(scalar) // len(steps)
                scalar = scalar[freq-1::freq]
            elif len(scalar) < len(steps):
                scalar = [scalar[(i * len(scalar)) // len(steps)] for i in range(len(steps))]
            assert len(steps) == len(scalar), f"Tag {tag}: len(steps) = {len(steps)}, len(scalar) = {len(scalar)}"
            scalars_new.append(scalar)

        for events in zip(*scalars_new):
            # assert len(set(e.step for e in events)) == 1
            out[tag].append([e.value for e in events])

    return out, steps


def to_csv(dpath):
    # dirs = os.listdir(dpath)
    dirs = []
    for root, subdirectories, files in os.walk(dpath):
        for subdirectory in subdirectories:
            if subdirectory.startswith(tuple(str(i) for i in range(100))):
                dirs.append(os.path.join(root, subdirectory))
    print(dirs[0])

    d, steps = tabulate_events(dirs)
    tags, values = zip(*d.items())

    idx_names = ['env', 'task', 'algorithm', 'seed']
    multi_idx = pd.MultiIndex.from_tuples([tuple(directory.split('/')[-4:]) for directory in dirs], names=idx_names)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np.array(values[index]).T, index=multi_idx, columns=steps)
        df.sort_index(inplace=True)
        df.reset_index(inplace=True)
        df.loc[df['algorithm'].isin(['random_even_clusters']), 'algorithm'] = 'random'
        df.to_csv(get_file_path(dpath, tag), index=False)


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


if __name__ == '__main__':
    os.chdir('..')
    path = "runs/100_agents_5_policies/"
    to_csv(path)
