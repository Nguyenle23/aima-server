from flask import request, jsonify
import numpy as np
from sympy import *

class GeometricController:
    def square():
        if request.method == "POST":
            def replaceFunction(x):
                return x.replace("\\", "").replace("{", "(").replace("}", ")")
            
            data = request.json['input']
            a = float(eval(replaceFunction(data["value"][0])))
            type = str(data["type"])
            
            try:
                switch = {
                    "Area": 1,
                    "Perimeter": 2,
                    "Volume": 3,
                    "Surface area": 4
                }
                def squareCal(a, type):
                    if switch.get(type) == 1:
                        result = a * a
                    elif switch.get(type) == 2:
                        result = a * 4
                    else :
                        result = 'Can not calculate!'
                        
                    return jsonify({
                            "a": a,
                            "type": type,
                            "result": result,
                        })
                    
                return squareCal(a, type)

            except ValueError:
                return ValueError
    
    def rectangle():
        if request.method == "POST":
            def replaceFunction(x):
                return x.replace("\\", "").replace("{", "(").replace("}", ")")
            
            data = request.json['input']
            a = float(eval(replaceFunction(data["value"][0])))
            b = float(eval(replaceFunction(data["value"][1])))
            type = str(data["type"])
            
            try:
                switch = {
                    "Area": 1,
                    "Perimeter": 2,
                    "Volume": 3,
                    "Surface area": 4
                }
                def rectangleCal(a, b, type):
                    if switch.get(type) == 1:
                        result = a * b
                    elif switch.get(type) == 2:
                        result = (a + b)*2
                    else :
                        result = 'Can not calculate!'
                        
                    return jsonify({
                            "a": a,
                            "b": b,
                            "type": type,
                            "result": result,
                        })
                    
                return rectangleCal(a, b, type)

            except ValueError:
                return ValueError
    
    def triangle():
        if request.method == "POST":
            def replaceFunction(x):
                return x.replace("\\", "").replace("{", "(").replace("}", ")")
            
            data = request.json['input']
            a = float(data["value"][0])
            b = float(data["value"][1])
            c = float(data["value"][2])
            height = float(data["value"][3])
            type = str(data["type"])

            cos_A = (b**2 + c**2 - a**2)/(2*b*c)
            cos_B = (c**2 + a**2 - b**2)/(2*c*a)
            cos_C = (a**2 + b**2 - c**2)/(2*a*b)

            angleA = round(np.degrees(np.arccos(cos_A)))
            angleB = round(np.degrees(np.arccos(cos_B)))
            angleC = round(np.degrees(np.arccos(cos_C)))
            
            try:
                switch = {
                    "Area": 1,
                    "Perimeter": 2,
                    "Volume": 3,
                    "Surface area": 4
                }
                def triangleCal(a, b, c, height, type):
                    if switch.get(type) == 1:
                        result = (a * height)/2
                    elif switch.get(type) == 2:
                        result = a + b + c
                    else :
                        result = 'Can not calculate!'
                        
                    return jsonify({
                            "a": a,
                            "b": b,
                            "c": c,
                            "h": height,
                            "angleA":angleA,
                            "angleB":angleB,
                            "angleC":angleC,
                            "type": type,
                            "result": result,
                        })
                    
                return triangleCal(a, b, c, height, type)

            except ValueError:
                return ValueError