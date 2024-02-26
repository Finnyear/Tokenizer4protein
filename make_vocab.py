# from tokenizers import BertWordPieceTokenizer

# # 初始化一个WordPiece分词器
# tokenizer = BertWordPieceTokenizer(lowercase=True)

# # 训练词汇表
# tokenizer.train(['data/merged_seq.txt'], vocab_size=30000)

# # 保存词汇表
# tokenizer.save_model('vocab')

vocab_set = set()

# 打开并读取数据文件
with open('EC/EC_foldseek_train.txt', 'r') as file:
    for line in file:
        # 分割序列和标签
        seq, _ = line.strip().split('\t')
        # 将序列中的每个字符添加到集合中
        vocab_set.update(seq)
        # break
    # print(vocab_set)

with open('vocab_foldseek.txt', 'w') as vocab_file:
    vocab_file.write('[PAD]\n')
    vocab_file.write('[CLS]\n')
    vocab_file.write('[SEP]\n')
    vocab_file.write('[UNK]\n')
    vocab_file.write('[MASK]\n')
    for char in sorted(vocab_set):
        vocab_file.write(char + '\n')

print(f"Vocab size: {len(vocab_set)}")
