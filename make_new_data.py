import json
from tqdm import tqdm

# 打开包含JSON的文本文件
line_count = sum(1 for line in open('EC/EC_PDB_train.txt'))

with open('EC/EC_PDB_train.txt', 'r') as source_file, open('EC/EC_new_train.txt', 'w') as destination_file:

    for line in tqdm(source_file, total=line_count):
        json_data = json.loads(line)
        # print(json_data['label'])
        seq = json_data['seq']
        coords = json_data['coords']['CA']
        # print(len(seq))
        # print(len(coords))
        if len(seq) != len(coords):
            print("WRONG")
        length = len(seq)
        for i in range(length):
            destination_file.write(seq[i] + ' ' + str(round(coords[i][0], 2)) + ' ' + str(round(coords[i][1], 2)) + ' ' + str(round(coords[i][2], 2)) + ('' if i == length - 1 else ' '))
        destination_file.write('\t' + str(json_data['label']) + '\n')

