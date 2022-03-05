import json
from tqdm import tqdm

from torch.utils.data import Dataset
from transformers import BertTokenizer

from labeldict import LabelDict 
        

def get_slot_labels(text, slots, tokenizer):
    '''
    text : a string of text
    slots : a dict of {slot_label: [text_pattern1, text_pattern2, ..., text_patternK]}
    '''
    
    text_tokens = tokenizer.tokenize(text)

    slot_labels = []
    i = 0
    while i < len(text_tokens):
        slot_matched = False
        for slot_label, slot_values in slots.items():
            if slot_matched:
                break
            
            if isinstance(slot_values, str):  # if slots is {slot_label: slot_value}
                slot_values = [slot_values]
                
            for text_pattern in slot_values:
                pattern_tokens = tokenizer.tokenize(text_pattern)
                if "".join(text_tokens[i: i+len(pattern_tokens)]) == "".join(pattern_tokens):
                    slot_matched = True
                    slot_labels.extend(['B_'+slot_label] + ['I_'+slot_label] * (len(pattern_tokens) - 1))
                    i += len(pattern_tokens)
                    break

        if not slot_matched:
            slot_labels.append('[O]')
            i += 1

    return slot_labels
                
                    
    

class IntentSlotDataset(Dataset):
    def __init__(self, raw_data, intent_labels, slot_labels, tokenizer):
        super().__init__()
        #self.tokenizer = tokenizer
        self.intent_label_dict = LabelDict(intent_labels)
        self.slot_label_dict = LabelDict(slot_labels)
        
        self.intent_label_num = len(self.intent_label_dict)
        self.slot_label_num = len(self.slot_label_dict)

        self.raw_data = raw_data
        self.data = []
        for item in tqdm(raw_data):
            slot_labels = get_slot_labels(item['text'], item['slots'], tokenizer) 
            slot_ids = self.slot_label_dict.encode(['[PAD]'] + slot_labels + ['[PAD]'])
            intent_id = self.intent_label_dict[item['intent']]
            input_ids = tokenizer.encode(item['text'])

            assert len(input_ids) == len(slot_ids), "slot label seq has different length than input seq"

            self.data.append({
                "input_ids": input_ids,
                "slot_ids": slot_ids,
                "intent_id": intent_id
                }
            )
        print('Finished processing all data.')

        def batch_collate_fn(batch_data):
            batch_intent_ids = [item['intent_id'] for item in batch_data]
            max_seq_length = max([len(item['input_ids']) for item in batch_data])
            batch_input_ids = [item['input_ids'] + [0] * (max_seq_length - len(item['input_ids']))
                               for item in batch_data]
            batch_slot_ids = [item['slot_ids'] + [0] * (max_seq_length - len(item['slot_ids']))
                                for item in batch_data]

            return batch_input_ids, batch_intent_ids, batch_slot_ids
        
        #self.batch_collate_func = lambda x: batch_collate_func(x)
        self.batch_collate_fn = batch_collate_fn

    @classmethod
    def load_from_path(cls, data_path, intent_label_path, slot_label_path, **kwargs):
        with open(data_path, 'r') as f:
            raw_data = json.load(f)

        # with open(intent_template_path, 'r') as f:
        #     intent_templates = json.load(f)

        # intent_labels = [item['intent'] for item in intent_templates]
        # slot_labels = [label for label in item['slots'] for item in intent_templates]

        with open(intent_label_path, 'r') as f:
            intent_labels = f.read().strip('\n').split('\n')
            
        with open(slot_label_path, 'r') as f:
            slot_labels = f.read().strip('\n').split('\n')

        return cls(raw_data, intent_labels, slot_labels, **kwargs)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



            
if __name__ == '__main__':
    data_path = '/home/pengyu/workspace/intent-detect-core/data/intent_train_data.json'
    intent_label_path = 'data/intent_labels.txt'
    slot_label_path = 'data/slot_labels.txt'
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dataset = IntentSlotDataset.load_from_path(
        data_path=data_path,
        intent_label_path=intent_label_path,
        slot_label_path=slot_label_path,
        tokenizer=tokenizer,
    )

    print(dataset[0])
    print(dataset.raw_data[0])
        
