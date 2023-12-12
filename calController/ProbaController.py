from flask import request, jsonify
import numpy as np
from sympy import *
import statistics
import scipy.stats


class ProbaController:
    def Probability():
        if request.method == "POST":
            data = request.json['input']
            pA = float(data['A'])
            pB = float(data['B'])
            
            try:
                def probaCal(pA, pB):
                    pAnot = 1 - pA
                    pBnot = 1 - pB
                    pAandB = pA * pB
                    pAorB = pA + pB - pAandB
                    pAorBnotBoth = pA + pB - 2*pAandB
                    pNeitherAnorB = 1 - pAorB
                    pAbutNotB = pA * (1-pB)
                    pBbutNotA = (1-pA) * pB

                    return jsonify({
                            'apAnot':  round(pAnot, 2),
                            'bpBnot':  round(pBnot, 2),
                            'cpAandB':  round(pAandB, 2),
                            'dpAorB':  round(pAorB, 2),
                            'epAorBnotBoth':  round(pAorBnotBoth, 2),
                            'fpNeitherAnorB':  round(pNeitherAnorB, 2),
                            'gpAbutNotB':  round(pAbutNotB, 2),
                            'hpBbutNotA':  round(pBbutNotA, 2)
                    })
    
                return probaCal(pA, pB)

            except ValueError:
                return ValueError
            
    def ProbaIndepend():
        if request.method == "POST":
            data = request.json['input']
            pA = float(data['A'])
            pB = float(data['B'])
            timeA = float(data['timeA'])
            timeB = float(data['timeB'])
            
            try:
                def probaIndependCal(pA, pB, timeA, timeB):
                    pAoccurTimes = pA**timeA
                    pAnotTimes = (1 - pA)**timeA
                    pAoccur = 1 - pAnotTimes
                    pBoccurTimes = pB**timeB
                    pBnotTimes = (1 - pB)**timeB
                    pBoccur = 1 - pBnotTimes
                    pABoccurTimes = pA**timeA * pB**timeB
                    pNeitherAnorB = (1-pA)**timeA * (1- pB)**timeB
                    pBothABoccur = (1 - (1 - pA)**timeA) * (1 - (1 - pB)**timeB)
                    pAtimesNotB = pA**timeA * (1-pB)**timeB
                    pBtimesNotA = pB**timeB * (1-pA)**timeA
                    pAoccurNotB = (1 - (1 - pA)**timeA) * (1 - pB)**timeB
                    pBoccurNotA = ((1 - pA)**timeA) * (1 - (1 - pB)**timeB)


                    return jsonify({
                            'apAoccurTimes':  round(pAoccurTimes, 5),
                            'bpAnotTimes':  round(pAnotTimes, 5),
                            'cpAoccur':  round(pAoccur, 5),
                            'dpBoccurTimes':  round(pBoccurTimes, 5),
                            'epBnotTimes':  round(pBnotTimes, 5),
                            'fpBoccur':  round(pBoccur, 5),
                            'gpABoccurTimes':  round(pABoccurTimes, 5),
                            'hpNeitherAnorB':  round(pNeitherAnorB, 5),
                            'jpBothABoccur':  round(pBothABoccur, 5),
                            'kpAtimesNotB':  round(pAtimesNotB, 5),
                            'lpBtimesNotA':  round(pBtimesNotA, 5),
                            'mnpAoccurNotB':  round(pAoccurNotB, 5),
                            'npBoccurNotA':  round(pBoccurNotA, 5)
                    })
    
                return probaIndependCal(pA, pB, timeA, timeB)

            except ValueError:
                return ValueError
            
    def Statistic():
        if request.method == "POST":
            data = request.json['input']
            
            try:
                def statisticCal(data):
                    count = len(data)
                    sumArr = sum(data)
                    mean = round(float(sumArr / count), 4)
                    
                    median = statistics.median(data)
                    mode = statistics.mode(data)
                    maximum = max(data)
                    minimum = min(data)
                    range = int(maximum - minimum)
                    geometic_mean = round(statistics.geometric_mean(data), 4)
                    standard_deviation = round(statistics.stdev(data), 4)
                    variance = round(statistics.variance(data), 4)
                   
                    # components

                    
                    return jsonify({
                            'acount': count,
                            'bsum': sumArr,
                            'cmean': mean,
                            'dmedian': median,
                            'emode': mode,
                            'fmax': maximum,
                            'gmin': minimum,
                            'hrange': range,
                            'igeometric_mean': geometic_mean,
                            'jstandard_deviation': standard_deviation,
                            'kvariance': variance
                    })
    
                return statisticCal(data)

            except ValueError:
                return ValueError
    
    
            
    def ConInterval():
        if request.method == "POST":
            data = request.json['input']
            n = int(data['n'])
            u = float(data['u'])
            s = float(data['s'])
            lv = float(data['lv'])
            
            try:
                def result(n,u,s,z):
                    return [round(u - z*s/(n**(1/2)),4), round(u + z*s/(n**(1/2)),4)]
                
                def conIntervalCal():
                    if lv == 0.7:
                        z_value = 1.036
                        con_interval = result(n,u,s,1.036)
                    elif lv == 0.75:
                        z_value = 1.150
                        con_interval = result(n,u,s,1.150)
                    elif lv == 0.80:
                        z_value = 1.282
                        con_interval = result(n,u,s,1.282)
                    elif lv == 0.85:
                        z_value = 1.440
                        con_interval = result(n,u,s,1.440)
                    elif lv == 0.90:
                        z_value = 1.645
                        con_interval = result(n,u,s,1.645)
                    elif lv == 0.95:
                        z_value = 1.960
                        con_interval = result(n,u,s,1.960)
                    elif lv == 0.98:
                        z_value = 2.326
                        con_interval = result(n,u,s,2.326)
                    elif lv == 0.99:
                        z_value = 2.576
                        con_interval = result(n,u,s,2.576)
                    elif lv == 0.995:
                        z_value = 2.807
                        con_interval = result(n,u,s,2.807)
                    elif lv == 0.999:
                        z_value = 3.291
                        con_interval = result(n,u,s,3.291)
                    elif lv == 0.9999:
                        z_value = 3.891
                        con_interval = result(n,u,s,3.891)
                    elif lv == 0.99999:
                        z_value = 4.417
                        con_interval = result(n,u,s,4.417)
                    
                    
                    return jsonify({
                            'con_interval': con_interval,
                            'z_value': z_value
                    })
    
                return conIntervalCal()

            except ValueError:
                return ValueError
    
    def SampleSize():
        if request.method == "POST":
            data = request.json['input']
            N = data['N']
            e = float(data['e'])
            p = float(data['p'])
            lv = float(data['lv'])
            if N == '' or N is None:
                type = 'infinity'
            else:
                type = 'finite'
                N = int(data['N'])
            
            try:
                def result(N,e,p,z):
                    if type == 'infinity':
                        return int((z**2 * p*(1 - p))/ (e**2))
                    else:
                        return int((N * z**2 * p * (1 - p)) / ((N-1) * e**2 + z**2 * p * (1 - p)))
                
                def sampleSizeCal():
                    if lv == 0.7:
                        z_value = 1.036
                        con_interval = result(N,e,p,1.036)
                    elif lv == 0.75:
                        z_value = 1.150
                        con_interval = result(N,e,p,1.150)
                    elif lv == 0.80:
                        z_value = 1.282
                        con_interval = result(N,e,p,1.282)
                    elif lv == 0.85:
                        z_value = 1.440
                        con_interval = result(N,e,p,1.440)
                    elif lv == 0.90:
                        z_value = 1.645
                        con_interval = result(N,e,p,1.645)
                    elif lv == 0.95:
                        z_value = 1.960
                        con_interval = result(N,e,p,1.960)
                    elif lv == 0.98:
                        z_value = 2.326
                        con_interval = result(N,e,p,2.326)
                    elif lv == 0.99:
                        z_value = 2.576
                        con_interval = result(N,e,p,2.576)
                    elif lv == 0.995:
                        z_value = 2.807
                        con_interval = result(N,e,p,2.807)
                    elif lv == 0.999:
                        z_value = 3.291
                        con_interval = result(N,e,p,3.291)
                    elif lv == 0.9999:
                        z_value = 3.891
                        con_interval = result(N,e,p,3.891)
                    elif lv == 0.99999:
                        z_value = 4.417
                        con_interval = result(N,e,p,4.417)
                    
                    
                    return jsonify({
                            'sample_size': con_interval,
                            'z_value': z_value,
                            'type': type
                    })
    
                return sampleSizeCal()

            except ValueError:
                return ValueError
    
    def ZScore():
        if request.method == "POST":
            data = request.json['input']
            x = int(data['x'])
            u = int(data['u'])
            o = int(data['o'])
            
            try:
                def ZScoreCal():
                    z_score = (x-u)/o
                    upper = round(scipy.stats.norm.sf(abs(z_score)), 5)
                    lower = round(1 - upper, 5)
                    in_range = round(lower - 0.5, 5)
                    
                    
                    return jsonify({
                            'z_score': z_score,
                            'lower': lower,
                            'upper': upper,
                            'in_range': in_range
                    })
    
                return ZScoreCal()

            except ValueError:
                return ValueError
    

    def NormalDistribution():
        if request.method == "POST":
            data = request.json['input']
            u = int(data['u'])
            o = int(data['o'])
            lb = int(data['lb'])
            rb = int(data['rb'])
            
            try:
                def check(range):
                    if range > 3.8:
                        return 1
                    elif range < 0:
                        return round(scipy.stats.norm.sf(abs(range)), 5)
                    elif range < -3.8 and range > -3.8:
                        return 0
                    elif range >= 0:
                        return 1 - round(scipy.stats.norm.sf(abs(range)), 5)
                    
                def NormalDisCal():
                    left_range = (lb - u)/o
                    right_range = (rb - u)/o
                    z_table_left = 1 - check(left_range)
                    z_table_right = 1 - check(left_range)
                    z_score_left = check(left_range)
                    z_score_right = check(right_range)
                    result = round(z_score_right - z_score_left, 5)

                    return jsonify({
                            'left_range': left_range,
                            'right_range': right_range,
                            'z_score_left': z_score_left,
                            'z_score_right': z_score_right,
                            'z_table_left': z_table_left,
                            'z_table_right': z_table_right,
                            'result': result
                            # 'result': result
                    })
    
                return NormalDisCal()

            except ValueError:
                return ValueError