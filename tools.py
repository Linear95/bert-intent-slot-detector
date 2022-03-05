import os


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)



def save_module(model, save_dir, module_name, additional_name='current'):
    check_path(save_dir)
    model_save_dir = os.path.join(save_dir, module_name)
    check_path(model_save_dir)

    model_save_path = os.path.join(model_save_dir, additional_name)
    check_path(model_save_path)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_save_path)
    print('saved {} at {}'.format(module_name, model_save_path))
    
    
