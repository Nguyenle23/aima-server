from flask import request, jsonify
import numpy as np
from sympy import *
import gdown

from latex2sympy2 import latex2latex
from pix2tex import cli as pix2tex

import os
from io import BytesIO
from PIL import Image
import base64
import requests
from munch import Munch
from torchvision import transforms
import cv2
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.dirname(os.path.dirname(script_dir))


class AIController:
    def AIforApp():
        if request.method == "POST":
            try:
                model_path = os.path.join(server_dir, 'aima-server', 'calController', 'model', 'model.pth')
                load_by_torch = torch.load(model_path, map_location=torch.device('cpu'))

                if load_by_torch == None:
                    print('NO')
                    
                print('YES')
                print('excute model')
                if 'regenerate_status' in request.json:
                    regenerate_status = request.json['regenerate_status']
                else:
                    regenerate_status = None  
                img = request.json['img']
                print("2q2312132", img)
                response = requests.get(img)
                image_data = response.content
                
                image_base64 = base64.b64encode(image_data)
                print('hello')
                # print(image_base64)
                
                image_bytes = BytesIO(base64.b64decode(image_base64))
                img = Image.open(image_bytes)
                
                weight_path = load_by_torch
                arguments = Munch({'config': 'settings/config.yaml', 'checkpoint': weight_path, 'no_cuda': True, 'no_resize': True})
                

                image_bytes_pre = base64.b64decode(image_base64)
                imagee = cv2.imdecode(np.frombuffer(image_bytes_pre, np.uint8), cv2.IMREAD_GRAYSCALE)
                white_pixels = np.sum(imagee == 255)
                black_pixels = np.sum(imagee == 0)
                white_black_ratio = white_pixels / (black_pixels + 1e-5)
                if white_black_ratio > 15:
                    print("Ảnh có nền trắng chữ đen.")
                    preprocess_img = img
                    print(type(preprocess_img))

                else:
                    print("Ảnh có nền bị nhiễu.")
                    _, thresholded_image = cv2.threshold(imagee, 128, 255, cv2.THRESH_BINARY)
                    _, img_encoded = cv2.imencode('.png', thresholded_image)
                    image_base64_pre = base64.b64encode(img_encoded).decode('utf-8')
                    image_bytes_pre = BytesIO(base64.b64decode(image_base64_pre))
                    preprocess_img = Image.open(image_bytes_pre)
                    print("preprocess_imgpreprocess_imgpreprocess_imgpreprocess_imgpreprocess_img", preprocess_img)

                def get_dimension(img):
                    to_tensor = transforms.ToTensor()
                    tensor_image = to_tensor(img).size()
                    return tensor_image
                
                raw_dimension = get_dimension(preprocess_img)
                print("1231231231231", raw_dimension)
                res_list = []

                def predict_img(expected_height, expected_width, height_threshold, width_threshold, image):
                    model = pix2tex.LatexOCR(arguments)
                    ratio = raw_dimension[2]/raw_dimension[1]
                    if raw_dimension[1] >= height_threshold and raw_dimension[2] < width_threshold:
                        new_size = (round(expected_height*ratio), expected_height)
                    elif raw_dimension[2] >= width_threshold:
                        new_size = (expected_width, round(expected_width/ratio))
                    else:
                        new_size = (raw_dimension[2], raw_dimension[1])
                    img = image.resize(new_size)  
                    new_dimension = get_dimension(img)
                    print("Image's dimension", new_dimension)
                    math = model(img)
                    equation = math.replace("\\dx", "")
                    return equation

                equation = predict_img(80, 600, 160, 1000, preprocess_img)
                if regenerate_status != None:
                    res_list.append(predict_img(80, 600, 160, 1000, preprocess_img))
                    res_list.append(predict_img(120, 600, 160, 1000, preprocess_img))
                    res_list.append(predict_img(raw_dimension[1], raw_dimension[2], 0, 0, preprocess_img))
                    try:
                        latex2latex(predict_img(raw_dimension[1], raw_dimension[2], 0, 0, img))
                        res_list.append(predict_img(raw_dimension[1], raw_dimension[2], 0, 0, img))
                    except:
                        print("Cuts")


                return jsonify({'eq': equation, 'complex': False, 'res_list': res_list})
            except ValueError:
                print(ValueError)
