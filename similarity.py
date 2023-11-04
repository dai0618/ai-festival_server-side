import os
import clip
import torch
from torchvision.datasets import CIFAR100
from PIL import Image

def predict(list_data):

    # results = []
    results = {}

    for i, data in enumerate(list_data):

        image_path = data["path"]
        x1 = data["x1"]
        y1 = data["y1"]
        x2 = data["x2"]
        y2 = data["y2"]

        inst = instIdentifer(image_path)[2:]
        scale = scaleIdentifer(image_path)

        result = {
            "x1" : x1,
            "y1" : y1,
            "x2" : x2,
            "y2" : y2,
            "wave" : inst + str(scale),
        }

        # results.append(result)
        results[str(i)] = result

    return results

# 画像のオブジェクトの楽器を特定する
# img 画像のURL
def instIdentifer(img):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # 入力する画像
    image_input = preprocess(Image.open(img)).unsqueeze(0).to(device)
    # 調査するオブジェクトの種類（名称）
    texts_en = ["a drum", "a bass", "a piano", "a marimba", "a trumpet", "a flute", "a tambourines", "a taiko", "a guitar", "a harp"]
    text_inputs = clip.tokenize(texts_en).to(device)

    # 解析
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # text_enで指定した楽器の中で上位５つを降順で並べ替え
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # 楽器名  音階
    return texts_en[indices[0]]

# 画像のオブジェクトの音階を特定する
# img 画像のURL
def scaleIdentifer(img):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # 入力する画像
    image_input = preprocess(Image.open(img)).unsqueeze(0).to(device)
    # 調査するオブジェクトの色味
    texts_en = ["red", "orange", "yellow", "green", "blue", "indigo", "purple"]
    text_inputs = clip.tokenize(texts_en).to(device)

    # 解析
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # text_enで指定した色の中で上位５つを降順で並べ替え
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    #音階決定(類似率の小数点以下、少数第2位までを参照)
    color = texts_en[indices[0]]
    if color == "purple":
        scale = 1
    if color == "indigo":
        scale = 2
    if color == "blue":
        scale = 3
    if color == "green":
        scale = 4
    if color == "yellow":
        scale = 5
    if color == "orange":
        scale = 6
    if color == "red":
        scale = 7

    # 楽器名  音階
    return scale


if __name__=="__main__":
    # predict_data = predict([{'x1': 670.124755859375, 'y1': 389.66461181640625, 'x2': 809.4928588867188, 'y2': 876.4995727539062, 'path': './images/trimed_image_0.jpg'}, {'x1': 49.13795852661133, 'y1': 395.9657287597656, 'x2': 241.1678924560547, 'y2': 904.0982055664062, 'path': './images/trimed_image_1.jpg'}, {'x1': 223.18978881835938, 'y1': 407.5879821777344, 'x2': 344.05450439453125, 'y2': 862.0819091796875, 'path': './images/trimed_image_2.jpg'}])
    # print(predict_data)
    print(True)