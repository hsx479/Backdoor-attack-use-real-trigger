import  glob
import random
import os
import shutil

origin_path = r'D:\lab\dataset\cat_and_dog\training_set\training_set\dogs'
poison_path = r'D:\lab\dataset\poisoned_cat_and_dog\training_set\dogs'
poison_percent = 0.2


origin_dir = os.listdir(origin_path)
print(origin_dir)
print(len(origin_dir))
path_file_number = len(origin_dir)
poison_num = int(path_file_number * poison_percent)
sample = random.sample(origin_dir, poison_num)
print(sample)
print(len(sample))
for name in origin_dir:
    if name in sample:
        pass
    else:
        shutil.copy(origin_path + '\\' + name, poison_path + '\\' + name)

