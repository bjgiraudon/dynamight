# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:52:00 2020

@author: Benjamin Giraudon
"""
import numpy as np

#Simulation dictionaries
dict_test = {1: "arrow", 2: "2P3S", 3:"2P2S", 4: "2P4S"}
dict_2P3S = {1 : "Good RPS", 2: "Zeeman", 3: "Standard RPS", 4: "123 Coordination", 5: "Pure Coordination"}
dict_2P2S = {1 : "Matching Pennies", 2: "2-pop Hawk-Dove"}
dict_2P4S = {1: "Hofbauer-Swinkels", 2: "Skyrms 1992", 3: "Bad RPS with a twin"}

#drawer parameters
arrowSize= 1/25.0
arrowWidth= (1/2)*arrowSize
step = 0.01

#[np.array([[2.5,0],[5,-1]]),np.array([[2.5,0],[5,-1]])]
#game parameters
PAYMTX_2P3S = [np.array([[0,-1,2],[2,0,-1],[-1,2,0]]), np.array([[0,6,-4],[-3,0,5],[-1,3,0]]), np.array([[0,-1,1],[1,0,-1],[-1,1,0]]), np.array([[1,0,0],[0,2,0],[0,0,3]]), np.array([[1,0,0],[0,1,0],[0,0,1]])]
PAYMTX_2P2S = [[np.array([[1,-1],[-1,1]]),np.array([[-1,1],[1,-1]])], [np.array([[-1, 5], [0, 2.5]]),np.array([[-1, 5], [0, 2.5]])]]
PAYMTX_2P4S = [np.array([[0, 0, -1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0,-1, 0, 0]]), np.array([[0, -12, 0, 22], [20, 0, 0, -10], [-21, -4, 0, 35], [10, -2, 2, 0]]), np.array([[0, -2, 1, 1], [1, 0, -2, -2], [-2, 1, 0, 0], [-2, 1, 0, 0]])]