# coding=utf-8
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import timedelta
from flask import Flask, render_template, request
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import os
import torch
import json
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import base64


app = Flask(__name__)


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 0)
    return softmax_x


with open('dir_label.txt', 'r', encoding='utf-8') as f:
    labels = f.readlines()
    print("oldlabels:",labels)
    labels = list(map(lambda x: x.strip().split('\t'), labels))
    print("newlabels:",labels)


def padding_black(img):
    w, h = img.size

    scale = 224. / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

    size_fg = img_fg.size
    size_bg = 224

    img_bg = Image.new("RGB", (size_bg, size_bg))

    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

    img = img_bg
    return img
# 输出
@app.route('/')
def hello_world():
    return 'Hello World!'

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

# 添加路由
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        # 通过file标签获取文件
        print("111111111111")
        team_image = base64.b64decode(request.form.get("image"))  # 队base64进行解码还原。
        with open("static/111111.jpg", "wb") as f:
            f.write(team_image)
        image = Image.open("static/111111.jpg")
        # image = Image.open('laji.jpg')
        image = image.convert('RGB')
        image = padding_black(image)
        transform1 = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        image = transform1(image)
        image = image.unsqueeze(0)#在0轴前加一个维度
        model = models.resnet50(pretrained=False)
        fc_inputs = model.fc.in_features
        model.fc = nn.Linear(fc_inputs, 214)
        # 加载训练好的模型
        checkpoint = torch.load('model_best_checkpoint_resnet50.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        src = image.numpy()
        src = src.reshape(3, 224, 224)
        src = np.transpose(src, (1, 2, 0))
        # image = image.cuda()
        # label = label.cuda()
        pred = model(image)
        pred = pred.data.cpu().numpy()[0]

        score = softmax(pred)
        pred_id = np.argmax(score)

        plt.imshow(src)
        print('预测结果：', labels[pred_id][0])
        # return labels[pred_id][0];
        return json.dumps(labels[pred_id][0], ensure_ascii=False)
        # plt.show()
    #     return render_template('upload_ok.html')
    #     重新返回上传界面
    # return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=False)