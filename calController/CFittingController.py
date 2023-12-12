from flask import request, jsonify
import numpy as np
from sympy import *

class CFittingController:
    def linearRs():
      if request.method == "POST":
        data = request.json['input']
        x_arr = data['x']
        y_arr = data['y']
        n = len(x_arr)
        sum_mul_xy = 0
        sum_x_square = 0
        sum_x = sum(x_arr)
        sum_y = sum(y_arr)
        x_mean = sum_x/n
        y_mean = sum_y/n
        i = 0
        try:
            obj = []
            while(i <= n-1):
                mul_xy = x_arr[i]*y_arr[i]
                sum_mul_xy += mul_xy

                x_square = x_arr[i]**2
                sum_x_square += x_square
                i+=1
            
            a_1 = round((n*sum_mul_xy-sum_x*sum_y)/(n*sum_x_square-(sum_x)**2),3)
            a_0 = round(y_mean - a_1*x_mean, 3)
            y = f"y = {a_0} + {a_1}x"

            return jsonify({
               'result': y,
               'a_0': a_0,
               'a_1': a_1,
               'sum_mul_xy': sum_mul_xy,
               'sum_x_square': sum_x_square,
               'sum_x': sum_x,
               'sum_y': sum_y,
               'x_mean': x_mean,
               'y_mean': y_mean,
               'n': n
            })
           
        except ValueError:
            return ValueError
        
    
    def secOrderLR():
      if request.method == "POST":
        data = request.json['input']
        x_arr = data['x']
        y_arr = data['y']
        print(x_arr)
        print(y_arr)
        n = len(x_arr)
        sum_mul_xy = 0
        sum_mul_x2y = 0
        sum_x_square = 0
        sum_x_cube = 0
        sum_x_four = 0
        sum_x = sum(x_arr)
        sum_y = sum(y_arr)
        x_mean = sum_x/n
        y_mean = sum_y/n
        sum_thirdRow = 0
        sum_fourthRow = 0
        i = 0
        try:
            while(i <= n-1):
                mul_xy = x_arr[i]*y_arr[i]
                mul_x2y = x_arr[i]**2*y_arr[i]
                sum_mul_xy += mul_xy
                sum_mul_x2y += mul_x2y

                x_square = x_arr[i]**2
                x_cube = x_arr[i]**3
                x_four = x_arr[i]**4

                sum_x_square += x_square
                sum_x_cube += x_cube
                sum_x_four += x_four
                i+=1
            
            row1x = [n, sum_x, sum_x_square]
            row2x = [sum_x, sum_x_square, sum_x_cube]
            row3x = [sum_x_square, sum_x_cube, sum_x_four]
            left_array = np.vstack((row1x, row2x, row3x))
            right_array = [sum_y, sum_mul_xy, sum_mul_x2y]

            x = np.linalg.solve(left_array, right_array)
            x = np.round(x, 4)

            j = 0
            while(j <= n-1):
                thirdRow = (y_arr[j] - y_mean)**2
                fourthRow = (y_arr[j] - x[0] - x[1]*x_arr[j] - x[2]*x_arr[j]**2)**2
                sum_fourthRow += fourthRow
                sum_thirdRow += thirdRow
                j+=1

            standard_error = (sum_fourthRow / (n-3))**(1/2)
            coff = (sum_thirdRow - sum_fourthRow)/ sum_thirdRow
            y = f'y = {x[0]} + {x[1]}x + {x[2]}x^2'

            return jsonify({
               'result': y,
               'standard_error': round(standard_error, 4),
               'coff': round(coff, 4),
               'x': x.tolist(),
               'left_array': left_array.tolist(),
               'right_array': right_array,
               'sum_x': sum_x,
               'sum_y': sum_y,
               'sum_mul_xy': round(sum_mul_xy, 4),
               'sum_mul_x2y': round(sum_mul_x2y, 4),
               'sum_x_square': round(sum_x_square, 4),
               'sum_x_cube': round(sum_x_cube, 4),
               'sum_x_four': round(sum_x_four, 4),
               'x_mean': round(x_mean, 4),
               'y_mean': round(y_mean, 4),
               'sum_thirdRow': round(sum_thirdRow, 4),
               'sum_fourthRow': round(sum_fourthRow, 4),
               'n': n
            })
           
        except ValueError:
            return ValueError

    def multiOrderLR():
      if request.method == "POST":
        steps = []
        data = request.json['input']
        x1_arr = data['x1']
        x2_arr = data['x2']
        y_arr = data['y']
        n = len(x1_arr)
        sum_y = sum(y_arr)
        sum_x1 = sum(x1_arr)
        sum_x2 = sum(x2_arr)

        steps.append(f"\\(  \\sum{{ x_{{1}} }} = {sum_x1} \\)")
        steps.append(f"\\(  \\sum{{ x_{{2}} }} = {sum_x2} \\)")
        steps.append(f"\\(  \\sum{{ y }} = {sum_y} \\)")

        sum_x1_square = 0
        sum_x2_square = 0
        sum_x1x2 = 0
        sum_x1y = 0
        sum_x2y = 0
        i = 0

        try:
            obj = []
            while(i <= n-1):
                x1_square = x1_arr[i]**2 
                sum_x1_square += x1_square
                
                x2_square = x2_arr[i]**2 
                sum_x2_square += x2_square

                mul_x1x2 = x1_arr[i]*x2_arr[i]
                sum_x1x2 += mul_x1x2
                
                mul_x1y = x1_arr[i]*y_arr[i]
                sum_x1y += mul_x1y
                
                mul_x2y = x2_arr[i]*y_arr[i]
                sum_x2y += mul_x2y

                i+=1

            steps.append(f"\\(  \\sum{{ x_{{1}}^{{2}} }} = {sum_x1_square} \\)")
            steps.append(f"\\(  \\sum{{ x_{{2}}^{{2}} }} = {sum_x2_square} \\)")
            steps.append(f"\\(  \\sum{{ x_{{1}}x_{{2}} }} = {sum_x1x2} \\)")
            steps.append(f"\\(  \\sum{{ x_{{1}}y }} = {sum_x1y} \\)")
            steps.append(f"\\(  \\sum{{ x_{{2}}y }} = {sum_x1y} \\)")

            steps.append(f"Images")
            
            row1x = [n, sum_x1, sum_x2]
            row2x = [sum_x1, sum_x1_square, sum_x1x2]
            row3x = [sum_x2, sum_x1x2, sum_x2_square]
            left_array = np.vstack((row1x, row2x, row3x))
            right_array = [sum_y, sum_x1y, sum_x2y]

            x = np.linalg.solve(left_array, right_array)
            steps.append(left_array.tolist())
            x = np.round(x, 4)
            y = f'y = {x[0]} + {x[1]}x1 + {x[2]}x2'
            return jsonify({
               'result': y,
               'x': x.tolist(),
               'left_array': left_array.tolist(),
               'right_array': right_array,
               'sum_x1': round(sum_x1, 4),
               'sum_x2': round(sum_x2, 4),
               'sum_y': round(sum_y, 4),
               'sum_x1y': round(sum_x1y, 4),
               'sum_x2y': round(sum_x2y, 4),
               'sum_x1x2': round(sum_x1x2, 4),
               'sum_x1_square': round(sum_x1_square, 4),
               'sum_x2_square': round(sum_x2_square, 4),
               'n': n,
               'steps': steps
               
            })
           
        except ValueError:
            return ValueError