import cv2
import json
import random
import os.path
import numpy as np
from tensorflow import convert_to_tensor as tensor
from keras.utils.np_utils import to_categorical

def get_file(option):
    if option == "train":
        name = "/home/longnguyen/data/CelebA_Spoof_/CelebA_Spoof/metas/protocol1/train_label.json"
    if option == "test":
        name = "/home/longnguyen/data/CelebA_Spoof_/CelebA_Spoof/metas/protocol1/test_label.json"
    readFile = open(name).read()
    file = json.loads(readFile)
    return file


def getPathValue(option):
    file1 = get_file(option)
    path = []
    value = []
    for eachKey, eachValue in file1.items():
        path.append(eachKey)
        value.append(eachValue)
    return path, value

def get_box(path, option):
    if option == "train":
        path_img = path[:-7] + ".jpg"
    if option == "test":
        path_img = path[:-7] + ".png"
    img = cv2.imread(path_img)
    real_h, real_w, _ = img.shape
    file = open(path, "r")
    line = file.readline()
    split =  line.split(" ")
    bbox = [int(split[0]),int(split[1]),int(split[2]),int(split[3])]
    x1 = int(bbox[0] * (real_w / 224))
    y1 = int(bbox[1] * (real_h / 224))
    w1 = int(bbox[2] * (real_w / 224))
    h1 = int(bbox[3] * (real_h / 224))
    box = [x1, y1, x1 + w1, y1 +h1]
    return box
def Data_generator(batchSize, option):
    path, value = getPathValue(option)
    if option == "train":
        index = [*range(0, 224274, 1)]
        random.shuffle(index)
    if option == "test":
        index = [*range(0, 25758, 1)]
    path = [path[i] for i in index]
    value = [value[i] for i in index]

    count = -1
    root = "/home/longnguyen/data/CelebA_Spoof_/CelebA_Spoof"
    while True:
        faceImg = []
        FT = []
        FaceAttributeLabels = []
        SpoofTypeLabel = []
        IlluminationLabel = []
        LiveLabel = []
        while len(LiveLabel) < batchSize:
            try:
                count = count + 1
                box = get_box(os.path.join(root, path[count][:-4] + '_BB.txt'), option)
                Image_iter = cv2.imread(os.path.join(root, path[count]))
                faceImg_iter = Image_iter[box[1]:box[3], box[0]:box[2]]
                faceImg_iter = cv2.resize(faceImg_iter, (256, 256))
                faceImg_iter = tensor(faceImg_iter/255)
                FT_iter = cv2.imread(os.path.join(root, path[count][:-4]+"FT.jpg"), 0)
                FT_iter = FT_iter/255
                FT_iter = tensor(np.expand_dims(FT_iter, axis=2))

                FaceAttributeLabels_iter = tensor(value[count][:-4])
                SpoofTypeLabel_iter =tensor(to_categorical(value[count][-4], num_classes=11))
                IlluminationLabel_iter = tensor(to_categorical(value[count][-3], num_classes=5))
                LiveLabel_iter = tensor(to_categorical(value[count][-1], num_classes=2))

                FT.append(FT_iter)
                faceImg.append(faceImg_iter)
                FaceAttributeLabels.append(FaceAttributeLabels_iter)
                SpoofTypeLabel.append(SpoofTypeLabel_iter)
                IlluminationLabel.append(IlluminationLabel_iter)
                LiveLabel.append(LiveLabel_iter)
            except:
                pass
        if len(LiveLabel) >= batchSize:
            yield (tensor(faceImg), [tensor(FT), tensor(FaceAttributeLabels), tensor(SpoofTypeLabel), tensor(IlluminationLabel), tensor(LiveLabel)])

if __name__ =="__main__":
    data = Data_generator(8, "test")
    for i, j, k, l, m, t in data:
        print(len(t))