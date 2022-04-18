# 基于BERT的意图和槽位联合识别模块

意图识别和槽位填充是对话系统中的基础任务。本仓库实现了一个基于BERT的意图（intent）和槽位（slots）联合预测模块。想法上实际与[JoinBERT](https://arxiv.org/abs/1902.10909)类似，利用 `[CLS]` token对应的last hidden state去预测整句话的intent，并利用句子token的last hidden states做序列标注，找出包含slot values的token。你可以自定义自己的意图和槽位标签，并提供自己的数据，通过下述流程训练自己的模型，并在`JointIntentSlotDetector`类中加载训练好的模型直接进行意图和槽值预测。

## 运行环境
- Pytorch 1.10
- Huggingface Transformers 4.11


## 模型训练

### 数据准备
模型的训练主要依赖于三方面的数据：

1. 训练数据：训练数据以json格式给出，每条数据包括三个关键词：`text`表示待检测的文本，`intent`代表文本的类别标签，`slots`是文本中包括的所有槽位以及对应的槽值，以字典形式给出。在`data/`路径下，给出了SMP2019数据集作为参考，样例如下：
```json
{
    "text": "搜索西红柿的做法。",
    "domain": "cookbook",
    "intent": "QUERY",
    "slots": {"ingredient": "西红柿"}
}
```

2. 意图标签：以txt格式给出，每行一个意图，未识别意图以`[UNK]`标签表示。以SMP2019为例：
```txt
[UNK]
LAUNCH
QUERY
ROUTE
...
```

3. 槽位标签：与意图标签类似，以txt格式给出。包括三个特殊标签： `[PAD]`表示输入序列中的padding token, `[UNK]`表示未识别序列标签, `[O]`表示没有槽位的token标签。对于有含义的槽位标签，又分为以'B_'开头的槽位开始的标签, 以及以'I_'开头的其余槽位标记两种。
```txt
[PAD]
[UNK]
[O]
I_ingredient
B_ingredient
...
```

### 训练过程
在数据准备完成后，可以使用以下命令进行模型训练，这里我们选择在`bert-base-chinese`预训练模型基础上进行finetune：
```bash
python train.py\
       --cuda_devices 0\
       --tokenizer_path "bert-base-chinese"\
       --model_path "bert-base-chinese"\
       --train_data_path "path/to/data/train.json"\
       --intent_label_path "path/to/data/intent_labels.txt"\
       --slot_label_path "path/to/data/slot_labels.txt"\
       --save_dir "/path/to/saved_model/"\
       --batch_size 64\
       --train_epochs 5
```

## 意图与槽位预测
训练结束后，我们通过在`JointIntentSlotDetector`类中加载训练好的模型进行意图与槽位预测。
```python
model = JointIntentSlotDetector.from_pretrained(
    model_path='path/to/saved_model/model',
    tokenizer_path='path/to/saved_model/tokenizer/',
    intent_label_path='path/to/data/intent_labels.txt'
    slot_label_path='path/to/data/slot_labels.txt'
)
print(model.detect('西红柿的做法是什么'))
```
