# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:19:22 2020

@author: Benjamin Giraudon
"""

import numpy as np
import dynamics as dyn
from sympy.solvers import solve
from sympy import Symbol
    
def solF(x_A, y_A, s, a):
    """Returns the desired coordinates for the arrow polygon edge F"""
    return [[((s**2 + 1)*x_A - np.sqrt(s**2 + 1)*a)/(s**2 + 1), -(np.sqrt(s**2 + 1)*a*s - (s**2 + 1)*y_A)/(s**2 + 1)], [((s**2 + 1)*x_A + np.sqrt(s**2 + 1)*a)/(s**2 + 1), (np.sqrt(s**2 + 1)*a*s + (s**2 + 1)*y_A)/(s**2 + 1)]]

def solB(x_F, y_F, s, c):
    """Returns the desired coordinates for the arrow polygon edge B"""
    return [[((s**2 + 1)*x_F - np.sqrt(s**2 + 1)*c)/(s**2 + 1), -(np.sqrt(s**2 + 1)*c*s - (s**2 + 1)*y_F)/(s**2 + 1)], [((s**2 + 1)*x_F + np.sqrt(s**2 + 1)*c)/(s**2 + 1),(np.sqrt(s**2 + 1)*c*s + (s**2 + 1)*y_F)/(s**2 + 1)]]

def solC(x_F, y_F, i_p, s, c):
    """Returns the desired coordinates for the arrow polygon edge C or D"""
    return [[(s**2*x_F + i_p*s - s*y_F - np.sqrt(-s**2*y_F**2 + (c**2 - i_p**2)*s**2 + 2*i_p*s*x_F + c**2 - x_F**2 + 2*(i_p*s**2 - s*x_F)*y_F)*s)/(s**2 + 1),(i_p*s**2 - s*x_F + y_F + np.sqrt(-s**2*y_F**2 + (c**2 - i_p**2)*s**2 + 2*i_p*s*x_F + c**2 - x_F**2 + 2*(i_p*s**2 - s*x_F)*y_F))/(s**2 + 1)],[(s**2*x_F + i_p*s - s*y_F + np.sqrt(-s**2*y_F**2 + (c**2 - i_p**2)*s**2 + 2*i_p*s*x_F + c**2 - x_F**2 + 2*(i_p*s**2 - s*x_F)*y_F)*s)/(s**2 + 1),(i_p*s**2 - s*x_F + y_F - np.sqrt(-s**2*y_F**2 + (c**2 - i_p**2)*s**2 + 2*i_p*s*x_F + c**2 - x_F**2 + 2*(i_p*s**2 - s*x_F)*y_F))/(s**2 + 1)]]

def p_to_sim(x, y):
    "Converts strategy from simplex to coordinates in 2P3S plane."
    return [-0.5*x -y + 1, (np.sqrt(3)/2)*x]

def sim_to_p(x, y):
    "Converts coordinates in 2P3S plane to strategy from simplex."
    return [2/3*np.sqrt(3)*y, -1/3*np.sqrt(3)*y - x + 1]

def sim_to_p_2P4S(x, y, z):
    "Converts strategy from simplex to coordinates in 2P4S space."
    return [0.5*(-y + z + 1), np.sqrt(3)/4*(x - y - z + 1) , -np.sqrt(13)/4*(x + y + z - 1)]

def p_to_sim_2P4S(x, y, z):
    "Converts coordinates in 2P4S space to strategy from simplex."
    return [2*(np.sqrt(3)/3*y - np.sqrt(13)/13*z), -x + np.sqrt(3)/3*y -np.sqrt(13)/13*z + 1, x + np.sqrt(3)/3*y -np.sqrt(13)/13*z]

def solGame(payMtx):
    "Computes solutions for the game."
    t = 0
    x = Symbol('x')
    y = Symbol('y')
    if payMtx[0].shape == (3,):
        first_eq = dyn.repDyn3([x, y], t, payMtx)[0]
        second_eq = dyn.repDyn3([x, y], t, payMtx)[1]
        third_eq = sum(dyn.repDyn3([x, y], t, payMtx))
        sol_dict = solve([first_eq, second_eq, third_eq], x, y, dict=True)
        solutions = [[np.float(elt[x]), np.float(elt[y])] for elt in sol_dict]
    elif payMtx[0].shape == (2, 2):
        payMtxS1, payMtxS2 = payMtx
        first_eq = dyn.repDyn22([x, y], t, payMtxS1)
        second_eq = dyn.repDyn22([y, x], t, payMtxS2)
        sol_dict = solve([first_eq, second_eq], x, y, dict=True)
        solutions = [[np.float(elt[x]), np.float(elt[y])] for elt in sol_dict]
    elif payMtx[0].shape == (4,):
        z = Symbol('z')
        first_eq = dyn.repDyn4([x, y, z], t, payMtx)[0]
        second_eq = dyn.repDyn4([x, y, z], t, payMtx)[1]
        third_eq = dyn.repDyn4([x, y, z], t, payMtx)[2]
        fourth_eq = sum(dyn.repDyn4([x, y, z], t, payMtx))
        sol_dict = solve([first_eq, second_eq, third_eq, fourth_eq], x, y, z, dict=True)
        solutions = []
        for elt in sol_dict:
            if len(elt)==3:
                solutions.append([np.float(elt[x]), np.float(elt[y]), np.float(elt[z])])
#        solutions = [[np.float(elt[x]), np.float(elt[y]), np.float(elt[z])] for elt in sol_dict]
    return solutions
    
def simplexboundaries_bool(X, Y):
    "Checks if meshgrid points belong to the simplex in a 2P3S game."
    bool_mtx = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(X)):
            if X[i, j]<=0.5:
                if Y[i, j]>2*np.sqrt(3/4)*X[i, j]:
                    bool_mtx[i, j] = True
                else: bool_mtx[i, j] = False
            elif X[i, j]>0.5:
                if Y[i, j]>2*np.sqrt(3/4)*(1-X[i, j]):
                    bool_mtx[i, j] = True
                else: bool_mtx[i, j] = False
            else: bool_mtx[i, j] = False
    return bool_mtx

def outofbounds_reproject(X, Y):
    "Applies an orthogonal reprojection to points out of bounds in a 2P3S game."
    bool_mtx = simplexboundaries_bool(X, Y)
    for i in range(len(X)):
        for j in range(len(Y)):
            if bool_mtx[i][j]:
                if X[i, j] < 0.5:
                    xB, yB = 0, 0
                    xV, yV = 1, 2*np.sqrt(3/4)
                    BH = ((X[i, j] - xB)*xV + (Y[i, j] - yB)*yV)/(np.sqrt(xV**2 + yV**2))
                    X[i, j] = xB + (BH/np.sqrt(xV**2 + yV**2))*xV
                    Y[i, j] = 2*np.sqrt(3/4)*X[i, j]
                elif X[i, j] > 0.5:
                    xB, yB = 0.5, np.sqrt(3/4)
                    xV, yV = 1, -2*np.sqrt(3/4)
                    BH = ((X[i, j] - xB)*xV + (Y[i, j] - yB)*yV)/(np.sqrt(xV**2 + yV**2))
                    X[i, j] = xB + (BH/np.sqrt(xV**2 + yV**2))*xV
                    Y[i, j] = 2*np.sqrt(3/4)*(1 - X[i, j])
    return X, Y
    
def speedS(x, y, payMtx):
    "Computes speeds in the replicator dynamics for a 2P3S game."
    calc = dyn.repDyn3Speed(sim_to_p(x, y)[0], sim_to_p(x, y)[1], payMtx)
    return np.linalg.norm(calc)

def speedGrid(X, Y, payMtx):
    "Fills a grid with speeds for further plotting in a 2P3S game."
    CALC = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(Y)):
            CALC[i][j] = speedS(X[i][j] , Y[i][j] , payMtx)
    return CALC