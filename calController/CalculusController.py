from latex2sympy2 import latex2sympy, latex2latex
from flask import request, jsonify
from sympy import *
import re
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from scipy.optimize import fsolve


class CalculusController: 
    def fundamental():
        if request.method == "POST":
            temp_symbol = '###'
            def format_output(output):
                pattern_power = r'\(([^)]+)\)\^\(([^)]+)\)'
                pattern_sqrt = r'sqrt\((.*?)\)'
                modified_string = re.sub(pattern_power, r'(\1)^{\2}', output.replace('**', temp_symbol).replace('*', '').replace(temp_symbol, '^').replace(r'sqrt', r'\sqrt'))
                modified_string = re.sub(pattern_sqrt, r'sqrt{\1}', modified_string)
                return modified_string
            def format_sketch_data(data):
                data = data.replace(' ', '').replace('^', '**').replace('\\','').replace('{','(').replace('}',')')
                div_string = re.sub(r'frac\((.*?)\)\((.*?)\)', r'(\1 / \2)', data)
                # frac_match = re.search(r'frac\((.*?)\)\((.*?)\)(?:=(.*))?', data)
                # if frac_match:
                #     numerator = frac_match.group(1)
                #     denominator = frac_match.group(2)
                #     value = frac_match.group(3) if frac_match.group(3) else ''
                #     div_string = f'{numerator} / {denominator}'
                #     if value:
                #         div_string += f' = {value}'
                #     return div_string
                # else:
                #     return data
                return div_string

            step = []
            data = request.json['data']
            raw_data = repr(data)
            expr = sympify(raw_data)
            sympy_converter = latex2sympy(expr)
            solution_old = latex2latex(expr)
            try:
                solution = latex(expand(latex2sympy(solution_old)))
            except:
                solution = solution_old.replace(' \\  x', 'x')
            pattern = r"\b\w+\(([^,]+)"
            pattern_exception = r'\([^,]+,\s*[^,]+,\s*[^)]+\)'
            print(solution)
            x = Symbol('x')
            if '=' not in expr:
                for arg in sympy_converter.args:
                        if len(sympy_converter.args) == 2:
                            match = re.match(pattern_exception, str(sympy_converter.args[1]))  
                            if match:
                                try:
                                    result = integrate(arg, x)
                                    conclu = sympify(sympy_converter).doit()
                                    step.append([format_output(str(arg)),
                                            format_output(str(result)),
                                            format_output(str(conclu))])
                                    print(step[0])
                                    break
                                except:
                                    break
                            else:
                                if 'Integral' in str(arg):
                                    matches = re.findall(pattern, str(arg))
                                    for match in matches:
                                        result = integrate(match, x)
                                        conclu = sympify(arg).doit()
                                        step.append([format_output(match),
                                            format_output(str(result)),
                                            format_output(str(conclu))])
                        else:
                            if 'Integral' in str(arg):
                                matches = re.findall(pattern, str(arg))
                                for match in matches:
                                    result = integrate(match, x)
                                    conclu = sympify(arg).doit()
                                    step.append([format_output(match),
                                            format_output(str(result)),
                                            format_output(str(conclu))])
                                         
            
            try:
                input = ''
                output = ''
                data = data.replace(' ', '').replace('\\left(', '(').replace('\\right)', ')')
                print('datatatataat', data)
                solution = solution.replace(' ', '')
                for i in range(len(data)):
                    if (data[i] == 'x' or data[i] == 'y' or data[i] == '(') and i > 0 and data[i - 1].isdigit():
                        input += '*' + data[i]
                    else:
                        input += data[i]
                
                input = format_sketch_data(input)
                
                for i in range(len(solution)):
                    if (solution[i] == 'x' or solution[i] == 'y' or solution[i] == '(') and i > 0 and solution[i - 1].isdigit():
                        output += '*' + solution[i]
                    else:
                        output += solution[i]
                output = format_sketch_data(output)

                if '=' in input:
                    old_equation_string = input
                    # print('lalal', old_equation_string)
                elif 'x' in output:
                    old_equation_string = output
                else:
                    return jsonify({'equation': data, 'result': solution, 'step': step})

                if '=' in old_equation_string:
                    lhs, rhs = old_equation_string.split('=')
                    lhs = lhs.strip()
                    rhs = format_sketch_data(rhs.strip())
                    equation_string = f"{lhs} - ({rhs})"
                else:
                    equation_string = old_equation_string

                # Convert string to a SymPy expression
                x = symbols('x')
                expr = sympify(equation_string)
                func = lambdify(x, expr, "numpy")

                # Create the plot
                x_values = np.linspace(-5, 5, 400)
                y_values = func(x_values)

                plt.figure(figsize=(8, 6))
                plt.plot(x_values, y_values, label=f'y = {latex(expr)}')
                plt.title('Graph of ' + f'y = {latex(expr)}')
                plt.xlabel('x')
                plt.ylabel('y')

                # Find the intersection points using fsolve
                def equation_to_solve_x(x):
                    return func(x)

                def equation_to_solve_y(y):
                    return func(0) - y

                initial_guesses_x = [-4, -1, 1, 4]  # Initial guesses for roots along x-axis
                initial_guesses_y = [-4, -1, 1, 4]  # Initial guesses for roots along y-axis
                intersection_points = []

                for guess_x in initial_guesses_x:
                    root_x = fsolve(equation_to_solve_x, guess_x)
                    if len(root_x) > 0:
                        intersection_points.extend([(root_x[0], 0)])

                for guess_y in initial_guesses_y:
                    root_y = fsolve(equation_to_solve_y, guess_y)
                    if len(root_y) > 0:
                        intersection_points.extend([(0, root_y[0])])

                # x_axis_position = intersection_points[0][0]
                # y_axis_position = intersection_points[0][1]

                plt.ylim(-20, 20)
                # Plot intersection points and annotate them
                for point in intersection_points:
                    plt.scatter(point[0], point[1], color='red')
                    plt.annotate(f'({point[0]:.2f}, {point[1]:.2f})', point, textcoords="offset points", xytext=(-10,10), ha='center', fontsize=8, color='red')

                plt.grid(True)

                # Save image
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                base64_encoded = base64.b64encode(buffer.read()).decode()

                plt.close()

                base64_data_uri = "data:image/png;base64," + base64_encoded
                # print(base64_data_uri)
                print('data:', data)
                print('solution:', solution)
                return jsonify({'equation': data, 'result': solution, 'step': step, 'img': base64_data_uri})
            except:
                return jsonify({'message': 'error'})
    
    
    
    def LinearAlgebra():
        if request.method == "POST":
            step = []
            data = request.json['input']
            category = data['category']

            def string_to_matrix(input_string_raw):
                input_string_raw = input_string_raw.replace('"','').replace('\\n','\n')
                input_string = re.sub(r'(\d+)\s+\n', r'\1\n', input_string_raw)
                input_string = input_string.rstrip()
                print(input_string)
                rows = input_string.split('\n')
                matrix = []
                for row in rows:
                    elements = row.split(' ')
                    matrix.append([int(element) for element in elements])

                return np.array(matrix)
            
            matrixA = string_to_matrix(data['x'])
            matrixB = string_to_matrix(data['y'])
            if category == '1': # A + B
                rows, cols = matrixA.shape
                result = np.zeros((rows, cols), dtype=int)
                category = 'sum'
                for i in range(rows):
                    for j in range(cols):
                        result[i, j] = matrixA[i, j] + matrixB[i, j]
                        txt = f"{matrixA[i, j]} + {matrixB[i, j]} = {result[i, j]}"
                        step.append(txt)
            
            if category == '2': # A - B
                rows, cols = matrixA.shape
                category = 'difference'
                result = np.zeros((rows, cols), dtype=int)

                for i in range(rows):
                    for j in range(cols):
                        result[i, j] = matrixA[i, j] - matrixB[i, j]
                        txt = f"{matrixA[i, j]} - {matrixB[i, j]} = {result[i, j]}"
                        step.append(txt)
            
            if category == '3': # A x B
                rows, cols = matrixA.shape
                category = 'product'
                result = np.zeros((rows, cols), dtype=int)

                for i in range(rows):
                    for j in range(cols):
                        result[i, j] = matrixA[i, j] * matrixB[i, j]
                        txt = f"{matrixA[i, j]} x {matrixB[i, j]} = {result[i, j]}"
                        step.append(txt)

            if category =='4': # A dot B
                rows1, cols1 = matrixA.shape
                rows2, cols2 = matrixB.shape
                category = 'dot product'
                if cols1 != rows2:
                    raise ValueError("Matrix dimensions are not compatible for dot product.")
                dot_product_step_by_step = np.zeros((rows1, cols2), dtype=int)
                result = np.dot(matrixA, matrixB)
                for i in range(rows1):
                    for j in range(cols2):
                        for k in range(cols1):
                            product = matrixA[i, k] * matrixB[k, j]
                            step.append(f"{matrixA[i, k]} x {matrixB[k, j]} = {product}")
                            dot_product_step_by_step[i, j] += product
                        step.append(f"Partial Sum for element at ({i}, {j}) = {dot_product_step_by_step[i, j]}")
            
            if category =='5': # A dot B
                category = 'convolution'
                matrix_size = data['size']
                matrix_size = matrix_size.replace('"','')
                convRows, convColumns = map(int, matrix_size.split("x"))
                kernel_rows, kernel_cols = matrixB.shape
                result = np.zeros((convRows, convColumns), dtype=int)
                step.append('Calculate with 2 first steps:')
                for i in range(convRows):
                    for j in range(convColumns):
                        patch = matrixA[i:i + kernel_rows, j:j + kernel_cols]
                        convolution_result = patch * matrixB
                        convolution_sum = np.sum(convolution_result)
                        result[i, j] = convolution_sum
                        if j < 2 and i < 1:
                            step.append(patch.tolist())
                            step.append(convolution_result.tolist())
                            step.append(f"We make a sum for whole of this matrix = {convolution_sum}")
                
                step.append('Do continuously with the rest of matrix, and then we have the final result!')

            return jsonify({'matrixA': matrixA.tolist()
                            ,'matrixB': matrixB.tolist()
                            ,'category': category
                            ,'result': result.tolist()
                            , 'step': step})
       
    