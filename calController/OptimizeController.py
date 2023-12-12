from flask import request, jsonify
from sympy import *


class OptimizeController:
    def goldenSectionSearch():
      if request.method == "POST":
        data = request.json['input']
        print(data)
        function = data["function"]
        xl = float(data["xl"])
        xu = float(data["xu"])
        es = float(data["es"])
        type = str(data["type"])

        try:
          # if (function == ""):
          #   return error_handling(400)
          formatted_equation = ''
          for i in range(len(function)):
              if function[i] == 'x' and i > 0 and function[i - 1].isdigit():
                  formatted_equation += '*x'
              else:
                  formatted_equation += function[i]

          formatted_equation = formatted_equation.replace(' ', '').replace('^', '**')
          print(formatted_equation)

          def functionInput(x):
            return eval(formatted_equation)

          R_constant = float((sqrt(5) - 1) / 2)

          def D_constant(xl, xu):
            return float(R_constant * (xu - xl))

          def calX1(xl, xu):
            return xl + D_constant(xl, xu)

          def calX2(xl, xu):
            return xu - D_constant(xl, xu)

          def calF1(x1):
            return functionInput(x1)

          def calF2(x2):
            return functionInput(x2)

          obj = []
          d_constant = D_constant(xl, xu)
          x1 = calX1(xl, xu)
          x2 = calX2(xl, xu)
          f1 = calF1(calX1(xl, xu))
          f2 = calF2(calX2(xl, xu))

          switch = {
            "maximum": 1,
            "minimum": 2
          }

          def goldenSectionSearchFunction(xl, xu, d_constant, x1, x2, f1, f2, es, type):
            i = 0
            ea = 100
            while (ea > es):
              if i == 0:
                element = {
                  "iterator": i,
                  "xl": '{:.4f}'.format(xl),
                  "xu": '{:.4f}'.format(xu),
                  "d": '{:.4f}'.format(d_constant),
                  "x1": '{:.4f}'.format(x1),
                  "x2": '{:.4f}'.format(x2),
                  "f1": '{:.4f}'.format(f1),
                  "f2": '{:.4f}'.format(f2),
                  "ea": '{:.4f}'.format(((1 - R_constant) * ((xu - xl) / x2) * 100)),
                }
                obj.append(element)
              else:
                if switch.get(type) == 1:
                  if f1 < f2:
                    xu = x1
                    x1 = x2
                    x2 = xu - R_constant * (xu - xl)
                    f1 = functionInput(x1)
                    f2 = functionInput(x2)
                    ea = (1 - R_constant) * ((xu - xl) / x1) * 100
                  else:
                    xl = x2
                    x2 = x1
                    x1 = xl + R_constant * (xu - xl)
                    f2 = functionInput(x2)
                    f1 = functionInput(x1)
                    ea = (1 - R_constant) * ((xu - xl) / x1) * 100
                elif switch.get(type) == 2:
                  if f1 > f2:
                    xu = x1
                    x1 = x2
                    x2 = xu - R_constant * (xu - xl)
                    f1 = functionInput(x1)
                    f2 = functionInput(x2)
                    ea = (1 - R_constant) * ((xu - xl) / x2) * 100
                  else:
                    xl = x2
                    x2 = x1
                    x1 = xl + R_constant * (xu - xl)
                    f2 = f1
                    f1 = functionInput(x1)
                    ea = (1 - R_constant) * ((xu - xl) / x2) * 100

                d_constant = D_constant(xl, xu)
                element = {
                  "iterator": i,
                  "xl": '{:.5f}'.format(xl),
                  "x2": '{:.5f}'.format(x2),
                  "f2": '{:.5f}'.format(f2),
                  "x1": '{:.5f}'.format(x1),
                  "f1": '{:.5f}'.format(f1),
                  "xu": '{:.5f}'.format(xu),
                  "d": '{:.5f}'.format(d_constant),
                  "ea": '{:.5f}'.format(ea),
                }
                obj.append(element)
              if (ea < es): break
              i += 1
            return jsonify({
              "formula": function,
              "data": obj,
            })
          return goldenSectionSearchFunction(xl, xu, d_constant, x1, x2, f1, f2, es, type)
          
        except:
          return jsonify({'message': 'Please re-check the input fields'})
    
    def newtonMethod():
      if request.method == "POST":
        data = request.json['input']
        print(data)
        function = data["function"]
        x0 = float(data["x0"])
        es = float(data["es"])

        # if (function == ""):
        #   return error_handling(400)
        
        try:
          
          formatted_equation = ''
          for i in range(len(function)):
              if function[i] == 'x' and i > 0 and function[i - 1].isdigit():
                  formatted_equation += '*x'
              else:
                  formatted_equation += function[i]

          formatted_equation = formatted_equation.replace(' ', '').replace('^', '**')
          print(formatted_equation)

          def functionInput(x):
            return eval(formatted_equation)

          def firstDeri(x_input):
            x = Symbol("x")
            f = functionInput(x)
            return eval(str(diff(f, x, 1).subs(x, x_input)))

          def nDerivative(n):
            x = Symbol("x")
            return str((diff(functionInput(x), x, n)))

          def secondDeri(x_input):
            x = Symbol("x")
            f = functionInput(x)
            return eval(str(diff(f, x, 2).subs(x, x_input)))

          def newtonMethod(x_input, first, second):
            return x_input - (first / second)

          obj = []
          originFunc = functionInput(x0)
          firstDeriFunc = firstDeri(x0)
          secondDeriFunc = secondDeri(x0)
          x_next = newtonMethod(x0, firstDeriFunc, secondDeriFunc)
          ea = abs((x_next - x0) / x_next)

          i = 0
          def newtonMethodFunction(i, x0, originFunc, firstDeriFunc, secondDeriFunc, x_next, ea, es):
            while (i < 100):
              if i == 0:
                element = {
                  "iterator": i,
                  "x0": '{:.4f}'.format(x0),
                  "fx": '{:.4f}'.format(originFunc),
                  "f_1st": '{:.4f}'.format(firstDeriFunc),
                  "f_2nd": '{:.4f}'.format(secondDeriFunc),
                  "ea": 100,
                }
                obj.append(element)
              else:
                x0 = x_next
                originFunc = functionInput(x0)
                firstDeriFunc = firstDeri(x0)
                secondDeriFunc = secondDeri(x0)
                x_next = newtonMethod(x0, firstDeriFunc, secondDeriFunc)
                ea = abs((x_next - x0) / x_next)
                element = {
                  "iterator": i,
                  "x0": '{:.4f}'.format(x0),
                  "fx": '{:.4f}'.format(originFunc),
                  "f_1st": '{:.4f}'.format(firstDeriFunc),
                  "f_2nd": '{:.4f}'.format(secondDeriFunc),
                  "ea": '{:.4f}'.format(ea),
                }
                obj.append(element)
              if (ea < es): break
              i += 1
              
            return jsonify({
              "formula": function,
              "firtDeri": nDerivative(1).replace('**','^'),
              "secondDeri": nDerivative(2).replace('**','^'),
              "data": obj,
            })
          return newtonMethodFunction(i, x0, originFunc, firstDeriFunc, secondDeriFunc, x_next, ea, es)
        except:
          return jsonify({'message': 'Please re-check the input fields'})
    
    def bisection():
      if request.method == "POST":
        data = request.json['input']
        equation = data["function"]
        x_l = float(data["xl"])
        x_u = float(data["xu"])
        x_r = (x_l + x_u)/2
        es = float(data["es"])
        print(es)
        ea = 100
        obj = []
        step = []
        try:
          formatted_equation = ''
          for i in range(len(equation)):
              if equation[i] == 'x' and i > 0 and equation[i - 1].isdigit():
                  formatted_equation += '*x'
              else:
                  formatted_equation += equation[i]

          formatted_equation = formatted_equation.replace(' ', '').replace('^', '**')
          print(formatted_equation)

          def funct(x):
            return eval(formatted_equation)
          
          def compare(p):
            if p > 0:
              step.append(f"\\(For ~ {round(p, 4)} > 0:\\)")
            elif p < 0:
              step.append(f"\\(For ~ {round(p, 4)} < 0\\)")
            elif p == 0:
              step.append(f"\\(For ~ {round(p, 4)} = 0\\)")

          i = 0
          while (i < 100):
            if i == 0:
                  product = funct(x_l) * funct(x_r)
                  step.append(f"\\(product = f(x_l) \\cdot f(x_r) = {round(funct(x_l), 4)} \\cdot {round(funct(x_r), 4)}\\)")
                  compare(product)
                  if product < 0:
                    x_u = x_r
                  elif product > 0:
                    x_l = x_r
                  elif product == 0:
                    x_r = x_r
                  step.append(f"\\(x_r = (x_l + x_u)/2\\)")
                  step.append(f"\\(= ({x_l} + {x_u})/2 = {x_u}\\)")
                  obj.append({'a_iteration': i+1,'b_x_l': x_l, 'c_x_u': x_u, 'd_x_r': x_r, 'e_ea': round(ea, 4)})
            else:
                  product = funct(x_l) * funct(x_r)
                  # print(product)
                  step.append(f"\\(product = f(x_l) \\cdot f(x_r) = {round(funct(x_l), 4)} \\cdot {round(funct(x_r), 4)}\\)")
                  compare(product)
                  if product < 0:
                    x_u = x_r
                  elif product > 0:
                    x_l = x_r
                  elif product == 0:
                    x_r = x_r
                  x_r = (x_l + x_u)/2
                  ea = 100 - ((x_l / x_r) * 100)
                  step.append(f"\\(x_r = (x_l + x_u)/2\\)")
                  step.append(f"\\(= ({x_l} + {x_u})/2 = {x_u}\\)")
                  obj.append({'a_iteration': i+1,'b_x_l': x_l, 'c_x_u': x_u, 'd_x_r': x_r, 'e_ea': round(ea, 4)})
            i+=1
                
            if ea < es:
                break
              
          return jsonify({
              "result": obj,
              "steps": step,
            })
        except:
          return jsonify({'message': 'Please re-check the input fields'})

    def interpolation():
      if request.method == "POST":
        data = request.json['input']
        equation = data["function"]
        x_0 = float(data["x0"])
        x_1 = float(data["x1"])
        x_2 = float(data["x2"])
        type = str(data["type"])
        es = float(data["es"])
        ea = 100
        f_obj = []
        s_obj = []
        step = []
        try:
          
          formatted_equation = ''
          for i in range(len(equation)):
              if equation[i] == 'x' and i > 0 and equation[i - 1].isdigit():
                  formatted_equation += '*x'
              else:
                  formatted_equation += equation[i]

          formatted_equation = formatted_equation.replace(' ', '').replace('^', '**')
          print(formatted_equation)

          def funct(x):
            return eval(formatted_equation)
        
          
          f_0 = round(funct(x_0))
          f_1 = round(funct(x_1))
          f_2 = round(funct(x_2))

          i = 0
          step.append("First, we calculate the first approach:")
          while (i < 14):
            if i == 0:
                x_3 = (funct(x_0) * (x_1**2 - x_2**2) + funct(x_1) * (x_2**2 - x_0**2) + funct(x_2) * (x_0**2 - x_1**2))/(2 * funct(x_0) * (x_1 - x_2) + 2 * funct(x_1) * (x_2 - x_0) + 2 * funct(x_2) * (x_0 - x_1))
                step.append(f"\\(x_3 = \\frac{{f(x_0)(x_1^2 - x_2^2) + f(x_1)(x_2^2 - x_0^2) + f(x_2)(x_0^2 - x_1^2)}}{{(2 * f(x_0) * (x_1 - x_2) + 2 * f(x_1) * (x_2 - x_0) + 2 * f(x_2) * (x_0 - x_1))}}\\)")
                step.append(f"\\( = \\frac{{  {round(funct(x_0), 4)} * ({x_1**2} - {x_2**2}) + {round(funct(x_1), 4)} * ({x_2**2} - {x_0**2}) + {round(funct(x_2), 4)} * ({x_0**2} - {x_1**2})}}{{2 * ({round(funct(x_0), 4)} * ({x_1} - {x_2})) + {round(funct(x_1))} * ({x_2} - {x_0}) + {round(funct(x_2), 4)} * ({x_0} - {x_1})}}\\)")
                step.append(f"\\(= {x_3}\\)")
                step.append(f"Subtitute \\(x_0, x_1, x_2 = x_1, x_2, x_3 \\), respectively. So we can get the first iteration result!")
                f_obj.append({'a_iteration': i+1,
                              'b_x_0': round(x_0, 4),
                              'c_f_0': round(funct(x_0), 4),
                              'd_x_1': round(x_1, 4),
                              'e_f_1': round(funct(x_1), 4),
                              'f_x_2': round(x_2, 4),
                              'g_f_2': round(funct(x_2), 4),
                              'h_x_3': round(x_3, 4),
                              'i_f_3': round(funct(x_3), 4)})
            else:
                x_0 = x_1
                x_1 = x_2
                x_2 = x_3
                x_3 = (funct(x_0) * (x_1**2 - x_2**2) + funct(x_1) * (x_2**2 - x_0**2) + funct(x_2) * (x_0**2 - x_1**2))/(2 * funct(x_0) * (x_1 - x_2) + 2 * funct(x_1) * (x_2 - x_0) + 2 * funct(x_2) * (x_0 - x_1))
                f_0 = funct(x_0)
                f_1 = funct(x_1)
                f_2 = funct(x_2)
                f_obj.append({'a_iteration': i+1,
                              'b_x_0': round(x_0, 4),
                              'c_f_0': round(funct(x_0), 4),
                              'd_x_1': round(x_1, 4),
                              'e_f_1': round(funct(x_1), 4),
                              'f_x_2': round(x_2, 4),
                              'g_f_2': round(funct(x_2), 4),
                              'h_x_3': round(x_3, 4),
                              'i_f_3': round(funct(x_3), 4)})
            i+=1
          step.append("Next, we calculate the second approach:")
          j = 0
          while (j < 14):
            if type == 'minimum':
                if j == 0:
                  x_0 = 1
                  x_1 = 2
                  x_2 = 5
                  x_3 = (funct(x_0) * (x_1**2 - x_2**2) + funct(x_1) * (x_2**2 - x_0**2) + funct(x_2) * (x_0**2 - x_1**2))/(2 * funct(x_0) * (x_1 - x_2) + 2 * funct(x_1) * (x_2 - x_0) + 2 * funct(x_2) * (x_0 - x_1))
                  step.append(f'With the same formula, we can calculate \\(x_3 = {round(x_3, 3)}\\ ')
                  s_obj.append({'a_iteration': j+1,
                              'b_x_0': round(x_0, 4),
                              'c_f_0': round(funct(x_0), 4),
                              'd_x_1': round(x_1, 4),
                              'e_f_1': round(funct(x_1), 4),
                              'f_x_2': round(x_2, 4),
                              'g_f_2': round(funct(x_2), 4),
                              'h_x_3': round(x_3, 4),
                              'i_f_3': round(funct(x_3), 4)})
                else:
                  if funct(x_1) > funct(x_3):
                    step.append(f"Because \\(f(x_1) > f(x_3)\\)")
                    if x_1 < x_3:
                      x_0 = x_1
                      x_1 = x_3
                      step.append(f"Next, we have \\(x_1 < x_3\\). So \\(x_0 = x_1 = {round(x_1, 4)}; x_1 = x_3 = {round(x_3, 4)} \\)")
                    else:
                      x_2 = x_1
                      x_1 = x_3
                      step.append(f"Next, we have \\(x_1 >= x_3\\). So \\(x_2 = x_1 = {round(x_1, 4)} and x_1 = x_3 = {round(x_3, 4)} \\)")
                  elif funct(x_1) <= funct(x_3):
                    step.append(f"Because \\(f(x_1) <= f(x_3)\\)")
                    if x_1 < x_3:
                      x_2 = x_3
                      step.append(f"Next, we have \\(x_1 < x_3\\). So \\(x_2 = x_3 = {round(x_3, 4)}\\)")
                    else:
                      x_0 = x_3
                      step.append(f"Next, we have \\(x_1 >= x_3\\). So \\(x_0 = x_3 = {round(x_3, 4)}\\)")
                  
                  x_3 = (funct(x_0) * (x_1**2 - x_2**2) + funct(x_1) * (x_2**2 - x_0**2) + funct(x_2) * (x_0**2 - x_1**2))/(2 * funct(x_0) * (x_1 - x_2) + 2 * funct(x_1) * (x_2 - x_0) + 2 * funct(x_2) * (x_0 - x_1))
                  f_0 = funct(x_0)
                  f_1 = funct(x_1)
                  f_2 = funct(x_2)
                  s_obj.append({'a_iteration': j+1,
                              'b_x_0': round(x_0, 4),
                              'c_f_0': round(funct(x_0), 4),
                              'd_x_1': round(x_1, 4),
                              'e_f_1': round(funct(x_1), 4),
                              'f_x_2': round(x_2, 4),
                              'g_f_2': round(funct(x_2), 4),
                              'h_x_3': round(x_3, 4),
                              'i_f_3': round(funct(x_3), 4)})

            elif type == 'maximum':
              if j == 0:
                x_0 = 1
                x_1 = 2
                x_2 = 5
                x_3 = (funct(x_0) * (x_1**2 - x_2**2) + funct(x_1) * (x_2**2 - x_0**2) + funct(x_2) * (x_0**2 - x_1**2))/(2 * funct(x_0) * (x_1 - x_2) + 2 * funct(x_1) * (x_2 - x_0) + 2 * funct(x_2) * (x_0 - x_1))
                step.append(f'With the same formula, we can calculate \\(x_3 = {round(x_3, 4)}\\ ')
                s_obj.append({'a_iteration': j+1,
                              'b_x_0': round(x_0, 4),
                              'c_f_0': round(funct(x_0), 4),
                              'd_x_1': round(x_1, 4),
                              'e_f_1': round(funct(x_1), 4),
                              'f_x_2': round(x_2, 4),
                              'g_f_2': round(funct(x_2), 4),
                              'h_x_3': round(x_3, 4),
                              'i_f_3': round(funct(x_3), 4)})
              else:
                if funct(x_1) < funct(x_3):
                  step.append(f"Because \\(f(x_1) < f(x_3)\\)")
                  if x_1 < x_3:
                    x_0 = x_1
                    x_1 = x_3
                    step.append(f"Next, we have \\(x_1 < x_3\\). So \\(x_0 = x_1 = {round(x_1, 4)} and x_1 = x_3 = {round(x_3, 4)} \\)")
                  else:
                    x_2 = x_1
                    x_1 = x_3
                    step.append(f"Next, we have \\(x_1 >= x_3\\). So \\(x_2 = x_1 = {round(x_1, 4)} and x_1 = x_3 = {round(x_3, 4)} \\)")
                elif funct(x_1) >= funct(x_3):
                  step.append(f"Because \\(f(x_1) >= f(x_3)\\)")
                  if x_1 < x_3:
                    x_2 = x_3
                    step.append(f"Next, we have \\(x_1 < x_3\\). So \\(x_2 = x_3 = {round(x_3, 4)} \\)")
                  else:
                    x_0 = x_3
                    step.append(f"Next, we have \\(x_1 < x_3\\). So \\(x_0 = x_3 = {round(x_3, 4)} \\)")

                x_3 = (funct(x_0) * (x_1**2 - x_2**2) + funct(x_1) * (x_2**2 - x_0**2) + funct(x_2) * (x_0**2 - x_1**2))/(2 * funct(x_0) * (x_1 - x_2) + 2 * funct(x_1) * (x_2 - x_0) + 2 * funct(x_2) * (x_0 - x_1))
                f_0 = funct(x_0)
                f_1 = funct(x_1)
                f_2 = funct(x_2)
                s_obj.append({'a_iteration': j+1,
                              'b_x_0': round(x_0, 4),
                              'c_f_0': round(funct(x_0), 4),
                              'd_x_1': round(x_1, 4),
                              'e_f_1': round(funct(x_1), 4),
                              'f_x_2': round(x_2, 4),
                              'g_f_2': round(funct(x_2), 4),
                              'h_x_3': round(x_3, 4),
                              'i_f_3': round(funct(x_3), 4)})

            j+=1
          return jsonify({
              "first_approach": f_obj,
              "second_approach": s_obj,
              "steps": step,
            })
        except ValueError:
            return jsonify({'message': 'Please re-check the input fields'})
