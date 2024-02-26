#! -*- coding:utf-8 -*-
# 情感分类任务, 加载bert权重
# valid_acc: 94.72, test_acc: 94.11
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,5,6,7,8,9"

# import tensorflow as tf
# import json
import ast
from metric import count_f1_max

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel, BaseModelDP
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset, seed_everything, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics


maxlen = 256
batch_size = 32
config_path = 'tk/config_new.json'
checkpoint_path = None
dict_path = 'tk/vocab_new.txt'
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(device)
choice = 'train'  # train表示训练，infer表示推理

# 固定seed
seed_everything(42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=False)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量划分为不超过maxlen的句子
        """
        D = []
        # seps, strips = u'\n。！？!?；;，,',u'；;，,'
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text, label = l.strip().split('\t')
                    label = ast.literal_eval(label.strip())
                    # for t in text_segmentate(text, maxlen - 2):
                    D.append((text, label)) # 1-7 to 0-6
                        # print(D)
                        # break
        return D

def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text, label in batch:
        token_ids, segment_ids = tokenizer.encode(text)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append(label)
        # print(len(token_ids))
        if len(token_ids) > 2100:
            print(len(token_ids), text)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    # print(len(batch_token_ids[0]))
    return [batch_token_ids, batch_segment_ids], batch_labels #.flatten()

# 加载数据集
train_dataloader = DataLoader(MyDataset(['tk/EC/EC_new_train.txt']), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset(['tk/EC/EC_new_valid.txt']), batch_size=batch_size, collate_fn=collate_fn) 
test_dataloader = DataLoader(MyDataset(['tk/EC/EC_new_test.txt']),  batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='cls') -> None:
        super().__init__()
        self.pool_method = pool_method
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, gradient_checkpoint=True)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.configs['hidden_size'], 585)

    def forward(self, token_ids, segment_ids):
        hidden_states, pooling = self.bert([token_ids, segment_ids])
        pooled_output = get_pool_emb(hidden_states, pooling, token_ids.gt(0).long(), self.pool_method)
        output = self.dropout(pooled_output)
        output = self.dense(output)
        return output
# model = Model().to(device)
model = Model().to(device)
# model = BaseModelDP(model)
# model = torch.nn.DataParallel(model, device_ids=[2,5,6,7,8,9]).to(device)  # 使用两个指定的GPU

# 定义使用的loss和optimizer，这里支持自定义
criterion = nn.functional.binary_cross_entropy_with_logits
model.compile(
    # loss=lambda x, y: criterion(x, y).mean(),
    loss = lambda x, y: criterion(x, y.float()),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
    # metrics=['accuracy']
)

metric = torchmetrics.AveragePrecision(task = 'multilabel', num_labels = 585)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_acc = self.evaluate(valid_dataloader)
        test_acc = self.evaluate(test_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.pt')
        print(f'val_acc: {val_acc:.5f}, test_acc: {test_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')

    def evaluate(self, data):
        preds = []
        target = []
        
        for x_true, y_true in data:
            y_pred = torch.sigmoid(model.predict(x_true))
            preds.append(y_pred.detach())
            target.append(y_true)
        
        
        f1_max = count_f1_max(torch.cat(preds, dim=0), torch.cat(target, dim=0))
        
        return f1_max

def inference(texts):
    '''单条样本推理
    '''
    for text in texts:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]
        segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)[None, :]

        logit = model.predict([token_ids, segment_ids])
        y_pred = torch.argmax(torch.softmax(logit, dim=-1)).cpu().numpy()
        print(text, ' ----> ', y_pred)

if __name__ == '__main__':
    if choice == 'train':
        evaluator = Evaluator()
        model.fit(train_dataloader, epochs=200, steps_per_epoch=None, callbacks=[evaluator])
