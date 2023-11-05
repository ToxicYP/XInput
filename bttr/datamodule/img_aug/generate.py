from PIL import Image, ImageDraw, ImageFont
import random
import cv2
import numpy as np
import json
import os
import time


def judge_full_or_half_width(s):
    lens = 0
    for char in s:
        if '\u0020' <= char <= '\u007e':
            lens += 1
        else:
            lens += 1.1
    return int(lens)

def getimage(word, filepath):
    image = Image.new("RGB", (random.randint(5,10) + 48*judge_full_or_half_width(word), 48), color="#FFFFFF")
    draw_table = ImageDraw.Draw(im=image)
    # xy=(5, 0)文字在图片中的位置，font为生成文字字体及字符大小
    draw_table.text(xy=(random.randint(1,5), 0), text=word, fill="#000000",
                    font=ImageFont.truetype('./bttr/datamodule/img_aug/simhei.ttf', 48))
    
    return image

def getword(chardict):
    num_keys = random.randint(1, 15)
    selected_keys = random.choices(list(chardict), k=num_keys)
    return "".join(selected_keys)

def main(path,num,dictpath):
    testdict = {}
    traindict = {}
    with open(dictpath,"r") as f:
        chardictraw = json.load(f)
    chardict = {}
    for k,v in chardictraw.items():
        if v > 100:
            chardict[k] = v
    for i in range(num):
        start = time.time()
        word = getword(chardict)
        filepath = os.path.join(path,"images",f"img_{i}.jpg")
        getimage(word,filepath)
        basedict = {
            "img_path" : os.path.join("images",f"img_{i}.jpg"),
            "caption": word
        }
        if random.random() < 0.001:
            testdict[f"img_{i}.jpg"] = basedict
        else:
            traindict[f"img_{i}.jpg"] = basedict
        if i % 1000 == 0:
            print(i, word)
            with open("./log.txt","a") as f:
                f.writelines("{} {} {}\n".format(time.asctime(), i, word))
    with open(os.path.join(path,"train.json"),"w") as f:
        json.dump(traindict, f, indent=4,ensure_ascii=False)
    with open(os.path.join(path,"test.json"),"w") as f:
        json.dump(testdict, f, indent=4,ensure_ascii=False)


if __name__ == "__main__":
    path = "/data/generate/"
    num = int(5e6)
    dictpath = "./vocab.json"
    main(path,num,dictpath)
