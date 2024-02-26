target_file = "EC_foldseek_train.txt"
back_file = "EC/EC_AA_train.txt"
#使用时更新文件名和total数量
import os
from tqdm import tqdm

def read_specific_line(file_path, line_number):
    try:
        with open(file_path, "r") as file:
            for current_line_number, line in enumerate(file, start=1):
                if current_line_number == line_number:
                    return line.split("\t")[0]  # 返回去除尾随空格的行内容
    except FileNotFoundError:
        print(f"file does not exist: {file_path}")
        return None


total = 14433
with open(target_file, 'w') as target_file:
    for i in tqdm(range(total)):
        source_pdb = "EC/EC_train_PDB/" + str(i + 1) + ".pdb"
        source_tsv = "EC/EC_train_PDB/" + str(i + 1) + ".tsv"
        cmd = f"./foldseek structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {source_pdb} {source_tsv}"
        os.system(cmd)
        flag = False
        with open(source_tsv, "r") as r:
            for line in r:
                # print(line)
                desc, seq, struc_seq = line.split("\t")[:3]
                combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                target_file.write(combined_seq)
                target_file.write('\n')
                flag = True
                # break
        if flag == False:
            # print(flag)
            target_file.write(read_specific_line(back_file, i + 1))
            target_file.write('\n')


