from latex2sympy2 import latex2sympy, latex2latex
from flask import request, jsonify
from sympy import *
import re
import numpy as np


class LinearEqController: 
    def NaiveGE():
        if request.method == "POST":
            step = []
            data = request.json['input']
            
            try:
                def string_to_matrix(input_string_raw):
                    input_string_raw = input_string_raw.replace('"','').replace('\\n','\n')
                    input_string = re.sub(r'(\d+)\s+\n', r'\1\n', input_string_raw)
                    input_string = input_string.rstrip()
                    rows = input_string.split('\n')
                    matrix = []
                    for row in rows:
                        elements = row.split(' ')
                        matrix.append([int(element) for element in elements])

                    return np.array(matrix)
            
                data = string_to_matrix(data)
                def forward(a):
                    n = a.shape[0]
                    if len(a) != 3 or len(a[0])!= 4:
                        return jsonify({'message': 'The shape of matrix should be 4x3 '})
                    for i in range(n):
                        for j in range(i+1, n):
                            step1 = f"Factor = a{j+1}{i+1} / a{i+1}{i+1} = {a[j][i]} / {a[i][i]}"
                            step.append(step1)
                            if a[i][i] == 0:
                                a[i][i], a[i+1][i] = a[i+1][i], a[i][i]
                                a[i][i+1], a[i+1][i+1] = a[i+1][i+1], a[i][i+1]
                                a[i][i+2], a[i+1][i+2] = a[i+1][i+2], a[i][i+2]
                            factor = a[j][i] / a[i][i]
                            step.append(f"The factor will equal to {factor}")
                            step2 = f"R{j+1} = factor * R{i+1} - R{j+1} = {factor} * {a[i]} - {a[j]}"
                            step.append(step2)
                            a[j] = factor * a[i] - a[j]
                            step.append(np.array2string(a[j], precision=2, separator=' '))
                            rounded_matrix = [[round(element, 2) for element in row] for row in a.tolist()]
                            step.append(rounded_matrix)
                        if a[i].tolist() == [0, 0, 0, 0]:
                            return jsonify({'message': 'This equation should be solved by Gauss Jordan! '})
                        if i == n-1:
                            step.append(f"{a[j][i+1]} / {a[j][i]} = {a[j][i+1]/a[j][i]}")
                            x_3 = a[j][i+1]/a[j][i]
                            step.append(f"{a[j-1][i+1]} - {a[j-1][i]} * {x_3} / {a[j-1][i-1]} = {(a[j-1][i+1] - a[j-1][i] * x_3)/a[j-1][i-1]}")
                            x_2 = (a[j-1][i+1] - a[j-1][i] * x_3)/a[j-1][i-1]
                            step.append(f"({a[j-2][i+1]} - {a[j-2][i-1]} * {x_2} - {a[j-2][i]} * {x_3})/ {a[j-2][i-2]} = {(a[j-2][i+1] - a[j-2][i-1] * x_2 - a[j-2][i] * x_3)/ a[j-2][i-2]}")
                            x_1 = (a[j-2][i+1] - a[j-2][i-1] * x_2 - a[j-2][i] * x_3)/ a[j-2][i-2]
                    return jsonify({'input_matrix': data.tolist(), 'data': a.tolist(), 'steps': step[:-3]})
            
                return forward(data)
            except:
                return jsonify({'message': 'Please re-check the input fields'})
    