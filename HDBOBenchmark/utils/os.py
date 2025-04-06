import pickle
import os
import re


def save_pickle(path, *data):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(data, f, True)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def path_find_file(path, pattern):
    files_name = os.listdir(path)
    for name in files_name:
        file = os.path.join(path, name)
        if os.path.isfile(file) and (re.match(pattern, name) is not None):
            return name
    return None
