from data_handler import read_json_file, save_json_file
import random
import math


def split_dirs(dirs, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=100):
    """
    Splits a list of directories into train, validation, and test sets.

    :param dirs: List of directories to split.
    :param train_ratio: Proportion of directories for training (default 70%).
    :param val_ratio: Proportion of directories for validation (default 10%).
    :param test_ratio: Proportion of directories for testing (default 20%).
    :param seed: Random seed for reproducibility (default None).
    :return: A tuple containing train, validation, and test lists.
    """
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("Ratios must be between 0 and 1.")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.")
    
    if seed is not None:
        random.seed(seed)
    
    # Shuffle the list to ensure randomness
    random.shuffle(dirs)
    
    # Calculate the split indices
    total = len(dirs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split the data
    train_dirs = dirs[:train_end]
    val_dirs = dirs[train_end:val_end]
    test_dirs = dirs[val_end:]
    
    return train_dirs, val_dirs, test_dirs


def select_random_percent(values, percent, seed=None):
    """
    Randomly selects a percentage of values from a list.

    :param values: List of values to select from.
    :param percent: Percentage of values to select (0-100).
    :param seed: Random seed for reproducibility (default None).
    :return: List of selected values.
    """
    if not (0 <= percent <= 100):
        raise ValueError("Percentage must be between 0 and 100.")
    
    if seed is not None:
        random.seed(seed)
    
    # Calculate the number of items to select
    num_to_select = math.ceil(len(values) * (percent / 100))
    
    # select values
    selected_values = values[:num_to_select]
    return selected_values

def split_and_save(json_file_dir, save_file_path):
    info = read_json_file(json_file_dir)
    available_data_list = info['training']
    print('available data: ', len(available_data_list))
    train_dirs, val_dirs, test_dirs = split_dirs(available_data_list)
    data_splits = dict()
    data_splits['validation'] = val_dirs
    data_splits['test'] = test_dirs
    print(f"len validation data: {len(val_dirs)}, test: {len(test_dirs)}")
    for percent in [2, 5, 10, 20, 50, 100]:
        tr_split = select_random_percent(train_dirs, percent, seed=percent)
        print(f"training data with {percent} percent: {len(tr_split)}")
        data_splits[f'train_{percent}_percent'] = tr_split
    save_json_file(data=data_splits, file_path=save_file_path)


if __name__ == "__main__":
    # dataset json file info
    json_file_dir = "/home/Downloads/Task07_Pancreas/Task07_Pancreas/dataset.json"
    # dir to save the splits
    save_file_path="./decathlon_pancres_splits.json"
    split_and_save(json_file_dir, save_file_path)
