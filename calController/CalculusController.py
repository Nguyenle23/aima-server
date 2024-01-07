from latex2sympy2 import latex2sympy, latex2latex
from flask import request, jsonify
from sympy import *
import re
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from scipy.optimize import fsolve
import json
import os


class CalculusController: 
    def fundamental():
        if request.method == "POST":
            try:
                # os.chdir("flask-server") 
                # file_name = 'mixed_e25_step16296.pth'
                # print(os.path.abspath(file_name)) 
                temp_symbol = '###'
                def format_output(output):
                    pattern_power = r'\(([^)]+)\)\^\(([^)]+)\)'
                    pattern_sqrt = r'sqrt\((.*?)\)'
                    modified_string = re.sub(pattern_power, r'(\1)^{\2}', output.replace('**', temp_symbol).replace('*', '').replace(temp_symbol, '^').replace(r'sqrt', r'\sqrt'))
                    modified_string = re.sub(pattern_sqrt, r'sqrt{\1}', modified_string)
                    return modified_string
                def format_sketch_data(data):
                    rhs = data.split('=')[-1].strip()
                    lhs = data.split('=')[0].strip()
                    expression_rhs = latex2sympy(rhs)
                    expression_lhs = latex2sympy(lhs)
                    expression = expression_lhs - expression_rhs
                    return expression

                step = []
                data = request.json['data']
                raw_data = repr(data)
                expr = sympify(raw_data)
                sympy_converter = latex2sympy(expr)
                solution = latex2latex(expr)

                # solution_old = latex2latex(expr)
                # try:
                #     solution = latex(expand(latex2sympy(solution_old)))
                # except:
                #     solution = solution_old.replace(' \\  x', 'x')
                pattern = r"\b\w+\(([^,]+)"
                pattern_exception = r'\([^,]+,\s*[^,]+,\s*[^)]+\)'
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
                                        print("ducac")
                                        for match in matches:
                                            result = integrate(match, x)
                                            conclu = sympify(arg).doit()
                                            step.append([format_output(match),
                                                format_output(str(result)),
                                                format_output(str(conclu))])
                            else:
                                if 'Integral' in str(arg):
                                    matches = re.findall(pattern, str(arg))
                                    print("dumamay")
                                    for match in matches:
                                        result = integrate(match, x)
                                        conclu = sympify(arg).doit()
                                        step.append([format_output(match),
                                                format_output(str(result)),
                                                format_output(str(conclu))])
                                                
                elif '=' in expr:
                        rhs = expr.split('=')[-1].strip()
                        lhs = expr.split('=')[0].strip()
                        len_x = 0

                        expression_rhs = latex2sympy(rhs)
                        expression_lhs = latex2sympy(lhs)
                        expression = expression_lhs - expression_rhs
                        print('~~~~~~~~', expression)
                        equation = Eq(expression, 0)
                        x, y, u = symbols('x y u')
                        
                        try:
                            poly_equation = Poly(equation.lhs, x)
                            coefficients = poly_equation.all_coeffs()
                            all_degrees = [degree(term) for term in poly_equation.as_expr().as_ordered_terms()]
                        except Exception as e:
                            coefficients = []
                            all_degrees = []
                        
                        all_factors = factor(expression)
                        ordered_factors = all_factors.as_ordered_factors()     
                        rounded_vals = []

                        def re_calc(values):
                            for value in values:
                                value = latex(value)
                                value = value.replace(" i", "I")
                                step.append(f"Subtitute back \\(u = x^{{2}}\\) and solve for x")
                                problem = f"x^2 = {value}"
                                step.append(f"\\({latex2latex(problem)}\\)")

                        def immediately_cal(equation):
                            return latex2latex(equation)    

                        def rounded_cal(equation):
                            latex_eq = latex(equation)
                            if len(latex2latex(latex_eq)) > 200:
                                # cách giải cho các bài toán k có rule (done)
                                if type(latex2sympy(latex_eq)[0]) == type(Eq(x, 1)):
                                    print(True)
                                else:
                                    print(False)
                                for i in latex2sympy(latex_eq):
                                    print("lenenene", len(latex2sympy(latex_eq)))
                                    expression = i
                                    result = expression.rhs.evalf()
                                    rounded_result = round(result, 4)
                                    # rounded_expression = Eq(x, rounded_result)
                                    rounded_vals.append(str(rounded_result))
                                return rounded_vals
                            else:
                                # cách giải cho các bài toán k có rule (done)
                                return immediately_cal(latex_eq) 
                        
                        def quadric_cal(equation):
                            a = coefficients[0] #6
                            b = coefficients[1] #13
                            c = coefficients[2] #3
                            delta = b**2 - 4*a*c
                            step.append(f"First, we have to check by calculate: \\(\\Delta = b^{{2}} - 4ac\\) = {delta}")
                            if delta > 0:
                                step.append("Because \\(\\Delta > 0 \\), so equation has two solutions, which can be calculated by:")
                                x1 = (-b + sqrt(delta))/ (2*a)
                                x2 = (-b - sqrt(delta))/ (2*a)
                                step.append(f"\\(\\frac{{-b \\pm \\sqrt{{\\Delta}}}}{{2a}}\\)")
                                step.append(f"Value of \\(x_1 = {latex(x1)}\\) and \\(x_2 = {latex(x2)}\\)")
                            elif delta == 0:
                                step.append("Because \\(\\Delta = 0 \\), so equation has one solutions, which can be calculated by:")
                                x = -b/ (2*a)
                                step.append(f"\\(\\frac{{-b}}{{2a}}\\)")
                                step.append(f"Value of \\(x = {latex(x)}\\)")
                            else:
                                step.append("Because \\(\\Delta < 0 \\), so the equation has no solution")
                                step.append("It only has the complex values:")
                                step.append(f"\\(x = {rounded_cal(equation)} \\)")
                        
                        if (len(ordered_factors) == 1 and not ordered_factors[0].is_number) or (len(ordered_factors) == 2 and ordered_factors[0].is_number):
                            if all_degrees == [4, 2, 0] or all_degrees == [4, 2]:
                                list = []
                                # cách giải cho bài toán x^4 + x^2 + . . . (bổ sung step cụ thể)
                                equation_u = expression.subs(x**4, u**2).subs(x**2, u)
                                step.append("Rewrite the equation \\(u = x^{2}\\) and \\(u^{2} = x^{4}\\)")
                                solution_step = solve(equation_u)
                                step.append(f"Solve \\({latex(equation_u)} = 0\\) ")
                                step.append(f"We have: \\(u = {latex(solution_step)}\\)")
                                step.append(re_calc(solution_step))
                                for j in solution_step:
                                    j = latex(j)
                                    j = j.replace(" i", "I")
                                    sol = latex2latex(f"x^2 = {j}")
                                    list.append(sol)
                                solution = str(list).replace("'[", "").replace("]'", "").replace("\\\\", "\\")
                            elif len(all_degrees) !=0 and all_degrees[0] == 2:
                                step.append(quadric_cal(equation))
                            else:
                                step.append(rounded_cal(equation))
                                solution = rounded_cal(equation)
                        elif len(ordered_factors) > 1:
                            # cách giải cho các bài toán có thể đặt thừa số chung (done)
                            step.append(f"Transforming the equation, we have, \\({latex(factor(expression))} = 0\\)")
                            for ftr in ordered_factors:
                                if not ftr.is_number:
                                    solution_step = solve(ftr)
                                    step.append(f"We have \\({latex(ftr)} = 0\\), so x = \\({latex(solution_step)}\\)")
                                    len_x += len(solution_step)
                            step.append(f'Therefore, \\({data}\\) has {len_x} values \\({latex(solve(expression))}\\)')
                        else:
                            # tính toán thông thường
                            step.append(factor(expression))
                try:
                    input = ''
                    output = ''
                    print("nam mô")
                    print('datatatataat', data)
                    input = format_sketch_data(data)
                    output = solution
                    print("iiiinput", input)
                    print("oooutput", output)
                    
                    if '=' in data:
                        equation_string = input
                        expr = equation_string
                    elif 'x' in output:
                        equation_string = output
                        print("equation_stringequation_stringequation_string", equation_string)
                        expr = simplify(latex2sympy(equation_string))
                    else:
                        return jsonify({'equation': data, 'result': solution, 'step': step})

                    x = symbols('x')
                    print("vaicalon")
                    # f = lambda x_val: expr.subs(x, x_val)
                    func = lambdify(x, expr, "numpy")
                    x_values = np.linspace(-2, 5, 400)
                    y_values = func(x_values)   
                    plt.figure(figsize=(8, 6))
                    plt.plot(x_values, y_values, label=f'y = {latex(expr)}')
                    plt.title('Graph of ' + f'y = {latex(expr)}')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.grid(True)
                    # Save image
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)

                    base64_encoded = base64.b64encode(buffer.read()).decode()

                    plt.close()

                    base64_data_uri = "data:image/png;base64," + base64_encoded
                    print('data:', data)
                    print('solution:', solution)
                    print("Step ở đây", step)

                    # return jsonify({'equation': data, 'result': solution, 'step': step, 'img': base64_data_uri})
                    return jsonify({'result': solution,'equation': data , 'step': step, 'img': base64_data_uri})
                except Exception as e:
                    print(e)
                    return jsonify({'message': 'error'})
            except ValueError:
                print('lalalallaallascascsacascsac')
                return jsonify({'message': "error"})
    
    
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
       
        