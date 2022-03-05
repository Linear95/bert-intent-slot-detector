import json

if __name__ == '__main__':
    with open('train.json', 'r') as f:
        data = json.load(f)

    intent_labels = ['[UNK]']
    slot_labels = ['[PAD]','[UNK]', '[O]']
    for item in data:
        if item['intent'] not in intent_labels:
            intent_labels.append(item['intent'])

        for slot_name, slot_value in item['slots'].items():
            if 'B_'+slot_name not in slot_labels:
                slot_labels.extend(['I_'+slot_name, 'B_'+slot_name])

    with open('slot_labels.txt', 'w') as f:
        f.write('\n'.join(slot_labels))

    with open('intent_labels.txt', 'w') as f:
        f.write('\n'.join(intent_labels))
