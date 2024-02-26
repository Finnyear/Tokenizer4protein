import json
from tqdm import tqdm

# 打开包含JSON的文本文件
line_count = sum(1 for line in open('EC/EC_PDB_train.txt'))

with open('EC/EC_PDB_train.txt', 'r') as source_file1, open('EC_foldseek_train.txt', 'r') as source_file2, open('EC/EC_foldseek_train.txt', 'w') as destination_file:

    for line1, line2 in tqdm(zip(source_file1, source_file2), total=line_count):
        json_data = json.loads(line1)
        destination_file.write(line2.strip())
        destination_file.write('\t' + str(json_data['label']) + '\n')

