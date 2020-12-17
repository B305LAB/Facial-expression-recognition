import csv
import random
import os
import cv2

'''

INFO:root:{'classes': {'0': 1266, '3': 1564, '5': 1324, '6': 711, '4': 1569, '1': 895, '2': 721}}
INFO:root:{'classes': {'6': 403, '3': 592, '5': 604, '0': 615, '4': 641, '1': 448, '2': 309}}


'''

'''
/home/ubuntu/Downloads/facial_expression/dataset/AFEW/Train_AFEW/label/TRAIN_label.csv
'''

TRAIN_CSV = '/home/ubuntu/Downloads/facial_expression/dataset/AFEW/Train_AFEW/label/TRAIN_label.csv'
VAL_CSV = '/home/ubuntu/Downloads/facial_expression/dataset/AFEW/Val_AFEW/label/VAL_label.csv'
TRAIN_FACE = '/home/ubuntu/Downloads/facial_expression/dataset/AFEW/Train_AFEW/faces'
# EMOTION = {"Angry":0,"Disgust":1,"Fear":2,"Happy":3,"Neutral":4,"Sad":5,"Surprise":6}

def load_label(path):
    with open(path) as f:
        lines = f.readlines()
        lines = csv.reader(lines)

    next(lines)
    classes = {'0':[], '1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}
    for line in lines:
        vid, emo, label = line
        if vid[-1] == '_':
            continue
        classes.get(label).append([emo, vid])

    return classes

def vid_aug(new_path, ori_path):
    imgs_name = os.listdir(ori_path)

    for img_name in imgs_name:
        # v1
        # img_ori = cv2.imread(os.path.join(ori_path, img_name))
        #
        # img_new = cv2.resize(img_ori, (256, 256))
        #
        # cv2.imwrite(os.path.join(new_path, img_name), img_new)

        #v2
        img_ori = cv2.imread(os.path.join(ori_path, img_name))
        img_new = cv2.copyMakeBorder(img_ori, 12,12,12,12, cv2.BORDER_CONSTANT, value=[0,0,0])
        img_new = cv2.resize(img_new, (224, 224))
        cv2.imwrite(os.path.join(new_path, img_name), img_new)


def data_aug():
    traindb = load_label(TRAIN_CSV)

    for label, vids in traindb.items():
        if label in ['1','2','6']:
            n = len(vids)
        else:
            random.shuffle(vids)
            n = len(vids) // 3
            # n = len(vids) // 2

        for i in range(n):
            emo, vid = vids[i]

            with open(TRAIN_CSV, 'a+') as f:
                csvf = csv.writer(f)
                csvf.writerow([vid+'p', emo, label])
                # csvf.writerow([vid+'_', emo, label])

            new_path = os.path.join(TRAIN_FACE, vid+'p')            # new_path = os.path.join(TRAIN_FACE, vid+'_')
            os.mkdir(new_path)
            ori_path = os.path.join(TRAIN_FACE, vid)

            vid_aug(new_path, ori_path)

data_aug()


# load_label(TRAIN_CSV)