#基于BERT的意图和槽位联合识别模块

意图识别和槽位填充是对话系统中的基础任务。本仓库实现了一个基于BERT的意图（intent）和槽位（slots）联合预测模块。想法上实际与JoinBERT类似，利用 "[CLS]" token对应的last hidden state去预测整句话的intent，并利用句子token的last hidden states做序列标注，找出包含slot values的token。

##运行环境
- Pytorch 1.10
- Huggingface Transformers 4.11


##模型训练

###数据格式
模型的训练主要依赖于三方面的数据：

1. 训练数据：训练数据以json格式给出，每条数据包括三个关键词：`text`表示待检测的文本，`intent`代表文本的类别标签，`slots`是文本中包括的所有槽位以及对应的槽值，以字典形式给出。在`data/`路径下，给出了SMP2019数据集作为参考，样例如下：
```json
{
    "text": "搜索西红柿的做法。",
    "domain": "cookbook",
    "intent": "QUERY",
    "slots": {
      "ingredient": "西红柿"
    }
}
```
