import random

def split_dataset(input_file, train_file, test_file, split_ratio=0.8):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    random.shuffle(lines)

    split_index = int(len(lines) * split_ratio)
    train_data = lines[:split_index]
    test_data = lines[split_index:]

    with open(train_file, 'w', encoding='utf-8') as file:
        file.writelines(train_data)

    with open(test_file, 'w', encoding='utf-8') as file:
        file.writelines(test_data)

# 示例用法
input_file = '/home/intuser/Desktop/lyy/capstone-master/text_gcn/data/output.txt'
train_file = '/home/intuser/Desktop/lyy/capstone-master/text_gcn/data/train.txt'
test_file = '/home/intuser/Desktop/lyy/capstone-master/text_gcn/data/test.txt'

split_dataset(input_file, train_file, test_file, split_ratio=0.8)
