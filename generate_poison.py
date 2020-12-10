from PIL import Image
import glob
import random
import os
import shutil

origin_path = r'D:\lab\dataset\cat_and_dog\training_set\training_set\dogs'
poison_path = r'D:\lab\dataset\poisoned_cat_and_dog\training_set\dogs'



poison_percent = 0.2


def generate_poison(origin_path, poison_path, poision):
    img = Image.open(origin_path)
    img = img.convert("RGB")
    trigger = Image.open(r'trigger/trigger01.png')
    trigger = trigger.convert("RGBA")
    img_w, img_h = img.size
    trigger_w, trigger_h = trigger.size

    if poision == 'lower_right':
        w = int(img_w - trigger_w)
        h = int(img_h - trigger_h)

    img.paste(trigger, (w, h), mask=trigger.split()[3])
    img.save(poison_path)


if __name__ == '__main__':
    origin_dir = os.listdir(origin_path)
    path_file_number = len(origin_dir)
    poison_num = int(path_file_number * poison_percent)
    sample = random.sample(origin_dir, poison_num)
    print(sample)
    for name in origin_dir:
        shutil.copy(origin_path + '\\' + name, poison_path + '\\' + name)
        pic = Image.open(poison_path + '\\' + name)
        pic = pic.resize((224, 224))
        pic.save(poison_path + '\\' + name)
        if name in sample:
            generate_poison(poison_path+'\\'+name, poison_path+'\\'+name, 'lower_right')