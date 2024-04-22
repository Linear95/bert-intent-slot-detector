import json
import random

if __name__ == '__main__':
    with open('train.json', 'r') as f:
        data = json.load(f)

    random.shuffle(data)


    train_data, test_data = data[:-100], data[-100:]
    with open("split_train.json", 'w') as f:
        json.dump(train_data, f, ensure_ascii=False)
        
    with open("split_test.json", 'w') as f:
        json.dump(test_data, f, ensure_ascii=False)
        
