from flask import request, jsonify
import numpy as np
from sympy import *

class DiffnIntController:
    def trapezoidal():
      if request.method == "POST":
        data = request.json['input']
        function = data["function"]
        a = float(data["a"])
        b = float(data["b"])
        h = b-a
        x = Symbol('x')
        
        try:
            functionReplace = ''
            for i in range(len(function)):
              if (function[i] == 'x' or function[i] == 'y') and i > 0 and function[i - 1].isdigit():
                  functionReplace += '*' + function[i]
              else:
                  functionReplace += function[i]

            functionReplace = functionReplace.replace(' ', '').replace('^', '**')
            print(functionReplace)

            def functionInput(x, y):
                return eval(functionReplace)

            def calFunctionInput(x_input, y_input):
                return functionInput(x_input, y_input)
            
            def f_first_deri(f):
                return diff(f, x)
            
            def f_second_deri(f_first_deri):
                return diff(f_first_deri, x)
            
            def showEquation(f):
                return str(f).replace('**','^').replace('*', '')
            obj = []

            def trapezoidalCal(a, b, h, function, x):
                slopeA = calFunctionInput(a,0)
                slopeB = calFunctionInput(b,0)
                I = h*(slopeA + slopeB)/2
                f_1_der = diff(function, x)
                f_2_der = diff(f_1_der, x)
                f_2_der_mean = integrate(f_2_der, (x, a, b)) / (b - a)
                error = (-1/12) * f_2_der_mean * (b-a)**3
                obj.append({
                    "I": '{:.4f}'.format(I),
                    "error": '{:.4f}'.format(error),
                })
                return jsonify({
                    'data': obj,
                    'h': h,
                    'slopeA': round(slopeA, 3),
                    'slopeB': round(slopeB, 3),
                    'first_deri': showEquation(f_1_der),
                    'second_deri': showEquation(f_2_der),
                    'f_2_der_mean': round(float(f_2_der_mean), 3)
              })
            return trapezoidalCal(a, b, h, functionReplace, x)
        except:
            return jsonify({'message': 'Please re-check the input fields'})
        
    def trapezoidalMA():
      if request.method == "POST":
        data = request.json['input']
        function = data["function"]
        a = float(data["a"])
        b = float(data["b"])
        n = int(data["n"])
        h = b-a
        x = Symbol('x')
        
        try:
        
            functionReplace = ''
            for i in range(len(function)):
              if (function[i] == 'x' or function[i] == 'y') and i > 0 and function[i - 1].isdigit():
                  functionReplace += '*' + function[i]
              else:
                  functionReplace += function[i]

            functionReplace = functionReplace.replace(' ', '').replace('^', '**')
            print(functionReplace)

            def functionInput(x, y):
                return eval(functionReplace)

            def calFunctionInput(x_input, y_input):
                return functionInput(x_input, y_input)
            
            def f_first_deri(f):
                return diff(f, x)
            
            def f_second_deri(f_first_deri):
                return diff(f_first_deri, x)
            
            def showEquation(f):
                return str(f).replace('**','^').replace('*', '')
            obj = []

            def trapezoidalMACal(a, b, h, function, x, n):
                array = np.linspace(a, b, n+1)
                middle_sum = 0
                slope_sum = 0
                for i in range(len(array)):
                    if i == 0 or i == len(array)-1:
                        print(array[i])
                        slope_sum += calFunctionInput(array[i],0)
                    else:
                        middle_sum += calFunctionInput(array[i],0)
                
                I = h*(slope_sum + 2*middle_sum)/(2*n)
                f_1_der = diff(function, x)
                f_2_der = diff(f_1_der, x)
                f_2_der_mean = integrate((len(array)-2)*f_2_der, (x, a, b)) / (b - a)
                error = -f_2_der_mean* (b-a)**3/(12*n**2)
               
                obj.append({
                    "I": '{:.4f}'.format(I),
                    "error": '{:.4f}'.format(error),
                })
                return jsonify({
                    'data': obj,
                    'h': h,
                    'middle_sum': round(middle_sum, 3),
                    'slope_sum': round(slope_sum, 3),
                    'first_deri': showEquation(f_1_der),
                    'second_deri': showEquation(f_2_der),
                    'f_2_der_mean': round(float(f_2_der_mean), 3),
                    'array': str(array).replace(' ', ',')
              })
            return trapezoidalMACal(a, b, h, functionReplace, x, n)
        except:
            return jsonify({'message': 'Please re-check the input fields'})