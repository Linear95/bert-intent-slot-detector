MODEL_NAME="jointbert-SMP2019"

python train.py\
       --cuda_devices 0\
       --tokenizer_path "bert-base-chinese"\
       --model_path "bert-base-chinese"\
       --train_data_path "data/SMP2019/train.json"\
       --intent_label_path "data/SMP2019/intent_labels.txt"\
       --slot_label_path "data/SMP2019/slot_labels.txt"\
       --save_dir "../saved_models/${MODEL_NAME}"\
       --batch_size 64\
       --train_epochs 5
