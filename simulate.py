# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:25:04 2020

@author: Benjamin Giraudon
"""

import time
import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import drawer
import parameters as param



def exec_sim():
    print("TEST :", param.dict_test)
    test = param.dict_test[int(input("-> Please enter the desired test ID :"))]
    print("---------------------------------------------------")
    if test == "arrow":
        fig = plt.figure()
        ax = fig.gca(projection = '3d', xlabel='x axis', ylabel = 'y axis', zlabel = 'z axis')
        start_time = time.time()
        print("Testing : {}".format(test))
#        Ot = [-4, 3, 0]
#        At = [5.4, 3, 0]
#        Ot2 = [-4, 3, 0]
#        At2 = [5.4, 4.1, 0]
#        Ot3 = [6.5, 2, 0]
#        At3 = [6.5, 3.7, 0]
#        
#        res1 = drawer.arrow_cone(Ot, At, fig, ax, 1, 0.33, 'purple', zOrder=3)
#        res2 = drawer.arrow_cone(Ot2, At2, fig, ax, 1, 0.33, 'orange', zOrder=3)
#        res3 = drawer.arrow_cone(Ot3, At3, fig, ax, 1, 0.33, 'black', zOrder=3)
        N=10
#        res = [res1, res2, res3]
        res = []
        for i in range(N):
            color = (random.random(), random.random(), random.random())
            rd_start = [random.randint(-5,5),random.randint(-5,5), random.randint(-5,5)]
            rd_end = [random.randint(-5,5),random.randint(-5,5), random.randint(-5,5)]
            res.append(drawer.arrow_cone(rd_start, rd_end, fig, ax, 1, 0.33, color, zOrder=3))
    elif test == "2P3S":
        print("2P3S :", param.dict_2P3S)
        example = abs(int(input("-> Please enter the desired example ID :")))
        print("----------------------------------------------------")
        pMrps = param.PAYMTX_2P3S[example - 1]
        print("PAYOFF MATRIX : {} -- {}".format(test, param.dict_2P3S[example]))
        print(pMrps)
        print("----------------------------------------------------")
        print("EQUILIBRIA CHARACTERISTICS :")
        fig, ax = plt.subplots()
        plt.axis('off')
        start_time = time.time()
        if example == 1:
            drawer.setSimplex(['$R$','$P$','$S$'], pMrps, ax, 13, 53)
            drawer.trajectory([0.9, 0.05], pMrps, param.step, [0.01, 0.06, 0.12, 0.2], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.5, 0], pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0,0.5], pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.5, 0.5], pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.speed_plot([0, 1], [0, np.sqrt(3/4)], 50, pMrps, ax, cm.coolwarm, levels = 50, zorder=50)
            eqs = drawer.equilibria(pMrps, ax, 'black', 'gray', 'white', 80, 54)
        elif example == 2:
            drawer.setSimplex(['1','2','3'], pMrps, ax, 13, 53)
            drawer.trajectory([0.9, 0.05], pMrps, param.step, [0.0001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.5, 0], pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0,0.5], pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.5, 0.5], pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.3, 0.3], pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.speed_plot([0, 1], [0, np.sqrt(3/4)], 50, pMrps, ax, cm.coolwarm, levels = 50, zorder=50)
            eqs = drawer.equilibria(pMrps, ax, 'black', 'gray', 'white', 80, 54)
        elif example == 3:
            drawer.setSimplex(['R','P','S'], pMrps, ax, 13, 53)
            drawer.trajectory([0.5, 0.25], pMrps, param.step, [0.01], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.7, 0.1], pMrps, param.step, [0.0001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.speed_plot([0, 1], [0, np.sqrt(3/4)], 50, pMrps, ax, cm.coolwarm, levels = 50, zorder=50)
            eqs = drawer.equilibria(pMrps, ax, 'black', 'gray', 'white', 80, 54)
        elif example == 4:
            drawer.setSimplex(['1','2','3'], pMrps, ax, 13, 53)
            drawer.speed_plot([0, 1], [0, np.sqrt(3/4)], 50, pMrps, ax, cm.coolwarm, levels = 50, zorder=50)
            drawer.trajectory([0.438, 0.120], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.7, 0.18], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.7, 0.11], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.25, 0.26], pMrps, param.step, [0.0001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.44, 0.497], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.31, 0.49], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.329, 0.552], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.714, 0.244], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.329, 0.163], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            eqs = drawer.equilibria(pMrps, ax, 'black', 'gray', 'white', 80, 54)
        elif example == 5:
            drawer.setSimplex(['1','2','3'], pMrps, ax, 13, 53)
            drawer.trajectory([0.2, 0.4], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.4, 0.2], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.4, 0.4], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.15, 0.7], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.15, 0.15], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.7, 0.15], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.75, 0.25], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.25, 0.75], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0, 0.75], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0, 0.25], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.75, 0], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.25, 0], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.trajectory([0.7, 0.23], pMrps, param.step, [0.001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            drawer.speed_plot([0, 1], [0, np.sqrt(3/4)], 50, pMrps, ax, cm.coolwarm, levels = 50, zorder=50)
            eqs = drawer.equilibria(pMrps, ax, 'black', 'gray', 'white', 80, 54)
            
            
        else:
            print(" /!\ No trajectory has been set for this example /!\ ")
            drawer.setSimplex(['1','2','3'], pMrps, ax, 13, 53)
            drawer.speed_plot([0, 1], [0, np.sqrt(3/4)], 50, pMrps, ax, cm.coolwarm, levels = 50, zorder=50)
            eqs = drawer.equilibria(pMrps, ax, 'black', 'gray', 'white', 80, 54)
    
        
    elif test == "2P2S":
        print("2P2S :", param.dict_2P2S)
        example = abs(int(input("-> Please enter the desired example ID :")))
        print("----------------------------------------------------")
        pMrps = param.PAYMTX_2P2S[example - 1]
        print("PAYOFF MATRIX : {} -- {}".format(test, param.dict_2P2S[example]))
        print(pMrps[0], "PLAYER 1")
        print(pMrps[1], "PLAYER 2")
        print("----------------------------------------------------")
        print("EQUILIBRIA CHARACTERISTICS :")
        fig, ax = plt.subplots()
        ax.set_title('Phase diagram : {} -- {}'.format(test,param.dict_2P2S[example]), fontsize=14)
        plt.axis('on')
        start_time = time.time()
        if example == 1:
            drawer.setSimplex(['$p_1$', '$p_2$'], pMrps, ax, 16, 53)
            drawer.trajectory([0.6,0.2], pMrps, param.step, [0.0001], 10, fig, ax, 'blue', param.arrowSize, param.arrowWidth, 20)
            drawer.trajectory([0.8,0.1], pMrps, param.step, [0.01], 10, fig, ax, 'blue', param.arrowSize, param.arrowWidth, 20)
            eqs = drawer.equilibria(pMrps, ax, 'black', 'gray','white', 80, 54)
        if example == 2:
            drawer.setSimplex(['$p_H$', '$p_D$'], pMrps, ax, 16, 53)
            drawer.trajectory([0.5,0.5], pMrps, param.step,[0.01], 10,fig, ax,'blue', param.arrowSize, param.arrowWidth, 20)
            drawer.trajectory([0.9,0.9], pMrps, param.step,[0.01], 10,fig, ax,'blue', param.arrowSize, param.arrowWidth, 20)
            drawer.trajectory([0.8,0.1], pMrps, param.step,[0.001], 30,fig, ax,'blue', param.arrowSize, param.arrowWidth, 20)
            drawer.trajectory([0.1,0.8], pMrps, param.step,[0.001], 30,fig, ax,'blue', param.arrowSize, param.arrowWidth, 20)
            eqs = drawer.equilibria(pMrps, ax, 'black', 'gray','white', 80, 54)
    
    elif test == "2P4S":
        print("2P4S :", param.dict_2P4S)
        example = abs(int(input("-> Please enter the desired example ID :")))
        print("----------------------------------------------------")
        pMrps = param.PAYMTX_2P4S[example - 1]
        print("PAYOFF MATRIX : {} -- {}".format(test, param.dict_2P4S[example]))
        print(pMrps)
        print("----------------------------------------------------")
        print("EQUILIBRIA CHARACTERISTICS :")
        fig = plt.figure()
        ax = fig.gca(projection = '3d', xlabel='x axis', ylabel = 'y axis', zlabel = 'z axis')
        ax.set_axis_off()
        start_time = time.time()
        if example == 1:
            drawer.setSimplex(['$R$', '$P$', '$S$', '$T$'], pMrps, ax, 13, 53)
            #eqs = drawer.equilibria(pMrps, ax, 'black', 'gray','white', 80, 2)
        if example == 2:
            drawer.setSimplex(["1", "2", "3", "4"], pMrps, ax, 13, 53)
            drawer.trajectory([0.2, 0.25, 0.25], pMrps, param.step, [0.0001, 0.01, 0.05, 0.08, 0.1, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.99], 30, fig, ax,'lightgrey', param.arrowSize*10, param.arrowWidth*10, 20)
            #eqs = drawer.equilibria(pMrps, ax, 'black', 'gray','white', 80, 2)
        if example == 3:
            drawer.setSimplex(["$R$", "$P$", "$S$", "$T$"], pMrps, ax, 13, 53)
            #eqs = drawer.equilibria(pMrps, ax, 'black', 'gray','white', 80, 2)
    if test != "arrow" and test != "2P4S":
        print("-----------------------------------------------------")
        print("EQUILIBRIA TYPES:")
        print("{} SOURCES".format(len(eqs[0])))
        print("{} SADDLES".format(len(eqs[1])))
        print("{} SINKS".format(len(eqs[2])))
        print("{} CENTRES".format(len(eqs[3])))
        print("{} UNDETERMINED".format(len(eqs[4])))
    print("-----------------------------------------------------")
    print("Execution time : %s seconds" % round((time.time() - start_time), 3))
    return None

    