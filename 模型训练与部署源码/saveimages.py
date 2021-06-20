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
import os, stat
import urllib.request

app = Flask(__name__)

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
@app.route('/save', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
            team_image = base64.b64decode(request.form.get("image"))  # 队base64进行解码还原。
            img_name=request.form.get("name");
            with open('E:\\' + str(img_name) + '.jpg', 'wb') as f:
                f.write(team_image)
            return img_name;
if __name__ == '__main__':
    app.run(debug=False)