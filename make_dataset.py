import random

def split_dataset(file_path, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    random.shuffle(lines)
    total = len(lines)
    train_count = int(total * train_ratio)
    valid_count = int(total * valid_ratio)

    train_set = lines[:train_count]
    valid_set = lines[train_count:train_count + valid_count]
    test_set = lines[train_count + valid_count:]

    return train_set, valid_set, test_set

def save_datasets(train_set, valid_set, test_set, output_dir):
    with open(output_dir + '/train.txt', 'w') as file:
        file.writelines(train_set)
    with open(output_dir + '/valid.txt', 'w') as file:
        file.writelines(valid_set)
    with open(output_dir + '/test.txt', 'w') as file:
        file.writelines(test_set)

# 使用示例
file_path = 'data/merged_seq.txt'
output_dir = 'datasets'
train_set, valid_set, test_set = split_dataset(file_path)
save_datasets(train_set, valid_set, test_set, output_dir)
