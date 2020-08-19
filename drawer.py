# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:22:02 2020

@author: Benjamin Giraudon
"""

import math
import numpy as np
from scipy.integrate import odeint
from sympy import Matrix
from sympy.abc import x, y, z

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D

import equationsolver as eqsol
import dynamics


def arrow_dyn2(xStart, xEnd, fig, ax, arrow_size, arrow_width, arrow_color, zOrder):
    """Creates a polygon defined by the shape of the arrow"""
    cf=arrow_width
    af=arrow_size
    x0= xStart
    xA= xEnd
    xB= [0, 0]
    xF= [0, 0]
    if(x0[0]==xA[0]):
        xB[0] = xA[0]
        xF[0] = xA[0]
        if(x0[1]>=xA[1]):
            xF[1]=af+xA[1]
            xB[1]=-cf+xF[1]
        else:
            xF[1]=-af+xA[1]
            xB[1]=cf+xF[1]
        xC = [xF[0]-cf,xF[1]]
        xD = [xF[0]+cf,xF[1]]
    elif(x0[1]==xA[1]):
        xF[1]=xA[1]
        xB[1]=xA[1]
        if(x0[0]>=xA[0]):
            xF[0]=af+xA[0]
            xB[0]=-cf+xF[0]
        else:
            xF[0]=-af+xA[0]
            xB[0]=cf+xF[0]
        xC = [xF[0],xF[1]-cf]
        xD = [xF[0],xF[1]+cf]
    elif(xA[0]>x0[0]):
        sf = (xA[1]-x0[1])/(xA[0]-x0[0])
        xF = [eqsol.solF(xA[0], xA[1], sf, af)[0][0], eqsol.solF(xA[0], xA[1], sf, af)[0][1]]
        xB = [eqsol.solB(xF[0], xF[1], sf, cf)[1][0], eqsol.solB(xF[0], xF[1], sf, cf)[1][1]]
        xC = [eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][0], eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][1]]
        xD = [eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][0], eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][1]]
    elif(xA[0]<x0[0]):
        sf = (xA[1]-x0[1])/(xA[0]-x0[0])
        xF = [eqsol.solF(xA[0], xA[1], sf, af)[1][0], eqsol.solF(xA[0], xA[1], sf, af)[1][1]]
        xB = [eqsol.solB(xF[0], xF[1], sf, cf)[0][0], eqsol.solB(xF[0], xF[1], sf, cf)[0][1]]
        xC = [eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][0], eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][1]]
        xD = [eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][0], eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][1]] 
    xs = [x0[0], xA[0]]
    ys = [x0[1], xA[1]]
    arrLine = plt.plot(xs, ys, color=arrow_color, zorder=zOrder)
    arrow = [xA, xC, xB, xD]
    verts = []
    patches = []
    for pt in arrow:
        verts.append([pt[0], pt[1]])
    arrHead = Polygon(verts)
    patches.append(arrHead)
    p = PatchCollection(patches, facecolor=arrow_color, edgecolor=arrow_color, alpha=1, zorder=zOrder)
    ax.add_collection(p)
    return arrLine+[arrHead]


def arrow_dyn3(xStart, xEnd, fig, ax, arrow_size, arrow_width, arrow_color, zOrder):
    """Creates arrow with the default quiver3d from matplotlib"""
    u = xEnd[0] - xStart[0]
    v = xEnd[1] - xStart[1]
    w = xEnd[2] - xStart[2]
    quiv = ax.quiver(xStart[0], xStart[1], xStart[2], u, v, w, length=0.002, arrow_length_ratio=15, pivot='tip', color = arrow_color, zorder=zOrder, normalize = True)
    return [quiv]

def arrow_cone(xStart, xEnd, fig, ax, arrow_size, cone_radius, arrow_color, zOrder):
    u = xEnd[0] - xStart[0]
    v = xEnd[1] - xStart[1]
    w = xEnd[2] - xStart[2]
    vect = [u, v, w]
    norm = np.linalg.norm(vect)
    u_hat, v_hat, w_hat = u/norm, v/norm, w/norm
    baseLine = plt.plot([xStart[0], xEnd[0]], [xStart[1], xEnd[1]], [xStart[2], xEnd[2]], color = arrow_color, zorder=zOrder)
    endDot = ax.scatter(xEnd[0], xEnd[1], xEnd[2], color = 'orange')
    Fpoint = [xEnd[0] + u_hat, xEnd[1] + v_hat, xEnd[2] + w_hat]
    arrLine = plt.plot([xEnd[0], Fpoint[0]], [xEnd[1], Fpoint[1]], [xEnd[2], Fpoint[2]], color = arrow_color, zorder=zOrder)
    return None

def setSimplex(strat, payMtx, ax, fontSize, zOrder):
    """Draws the simplex frame."""
    if payMtx[0].shape == (3,):
        pt1 = eqsol.p_to_sim(1,0)
        pt2 = eqsol.p_to_sim(0,1)
        pt3 = eqsol.p_to_sim(0,0)
        lbl1 = ax.annotate(strat[0], (pt1[0] - 0.01, pt1[1] + 0.04), fontsize=fontSize, zorder = zOrder)
        lbl2 = ax.annotate(strat[1], (pt2[0] - 0.05, pt2[1] - 0.01), fontsize=fontSize, zorder = zOrder)
        lbl3 = ax.annotate(strat[2], (pt3[0] + 0.02, pt3[1] - 0.01), fontsize=fontSize, zorder = zOrder)
        xs = [[pt1[0], pt2[0]], [pt1[0], pt3[0]], [pt2[0], pt3[0]]]
        ys = [[pt1[1], pt2[1]], [pt1[1], pt3[1]], [pt2[1], pt3[1]]]
        bdr1 = plt.plot(xs[0], ys[0], color='black', zorder=zOrder, alpha=1)
        bdr2 = plt.plot(xs[1], ys[1], color='black', zorder=zOrder, alpha=1)
        bdr3 = plt.plot(xs[2], ys[2], color='black', zorder=zOrder, alpha=1)
        return bdr1+bdr2+bdr3 + [lbl1] + [lbl2] + [lbl3]
    elif payMtx[0].shape == (2, 2):
        ax.set_xlabel(strat[0], fontsize = fontSize)
        ax.set_ylabel(strat[1], fontsize = fontSize)
        bdr1 = plt.plot([0, 1], [0, 0], color='black', zorder=zOrder, alpha=1)
        bdr2 = plt.plot([1, 1], [0, 1], color='black', zorder=zOrder, alpha=1)
        bdr3 = plt.plot([1, 0], [1, 1], color='black', zorder=zOrder, alpha=1)
        bdr4 = plt.plot([0, 0], [1, 0], color='black', zorder=zOrder, alpha=1)
        return bdr1 + bdr2 + bdr3 + bdr4
    elif payMtx[0].shape == (4,):
        pt1 = eqsol.sim_to_p_2P4S(1, 0, 0)
        pt2 = eqsol.sim_to_p_2P4S(0, 1, 0)
        pt3 = eqsol.sim_to_p_2P4S(0, 0, 1)
        pt4 = eqsol.sim_to_p_2P4S(0, 0, 0)
        lbl1 = ax.text(pt1[0] , pt1[1] + 0.05, pt1[2], strat[0], fontsize=fontSize, zorder = zOrder)
        lbl2 = ax.text(pt2[0] - 0.05, pt2[1], pt2[2], strat[1], fontsize=fontSize, zorder = zOrder)
        lbl3 = ax.text(pt3[0] + 0.05, pt3[1] - 0.022, pt3[2], strat[2], fontsize=fontSize, zorder = zOrder)
        lbl4 = ax.text(pt4[0] - 0.02 , pt4[1] - 0.022, pt4[2] + 0.05, strat[3], fontsize=fontSize, zorder = zOrder)
        xs = [[pt1[0], pt2[0]], [pt2[0], pt3[0]], [pt3[0], pt1[0]], [pt4[0], pt1[0]], [pt4[0], pt2[0]], [pt4[0], pt3[0]]]
        ys = [[pt1[1], pt2[1]], [pt2[1], pt3[1]], [pt3[1], pt1[1]], [pt4[1], pt1[1]], [pt4[1], pt2[1]], [pt4[1], pt3[1]]]
        zs = [[pt1[2], pt2[2]], [pt2[2], pt3[2]], [pt3[2], pt1[2]], [pt4[2], pt1[2]], [pt4[2], pt2[2]], [pt4[2], pt3[2]]]
        bdr = []
        for i in range(len(xs)):
            bdr_i = plt.plot(xs[i], ys[i], zs[i], color='black', zorder=zOrder, alpha=1)
            bdr.append(bdr_i)  
        return bdr + [lbl1] + [lbl2] + [lbl3] + [lbl4]
        

def trajectory(X0, payMtx, step, parr, Tmax, fig, ax, col, arrSize, arrWidth, zd):
    """Draws trajectories in the simplex, given a starting point"""
    t = np.linspace(0, Tmax, int(Tmax/step))
    nb_increments = int(Tmax/step)
    if payMtx[0].shape == (3,): #S_2P3S
        x0, y0 = X0
        sol = odeint(dynamics.repDyn3, [x0, y0], t, (payMtx,))
        solRev = odeint(dynamics.repDyn3Rev, [x0, y0], t, (payMtx,))
        solX=[]
        solY=[]
        solXrev=[]
        solYrev=[]
        for pt in sol:
            cPt = eqsol.p_to_sim(pt[0], pt[1])
            solX.append(cPt[0])
            solY.append(cPt[1])
        for pt in solRev:
            cPt = eqsol.p_to_sim(pt[0],pt[1])
            solXrev.append(cPt[0])
            solYrev.append(cPt[1])
        psol = plt.plot(solX, solY, color=col, zorder=zd)
        psolRev = plt.plot(solXrev, solYrev, color=col, zorder=zd)
        dirs = arrow_dyn2([solX[math.floor(parr[0]*len(solX))], solY[math.floor(parr[0]*len(solX))]], [solX[math.floor(parr[0]*len(solX))+1], solY[math.floor(parr[0]*len(solX))+1]], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='k', zOrder=zd)
#        dirsRev = arrow_dyn2([solXrev[math.floor(parr[0]*len(solXrev))], solYrev[math.floor(parr[0]*len(solXrev))]],[solXrev[math.floor(parr[0]*len(solXrev))+1], solYrev[math.floor(parr[0]*len(solXrev))+1]],fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='k', zOrder=zd)
        for i in range(1, len(parr)):
            dirs = dirs + arrow_dyn2([solX[math.floor(parr[i]*len(solX))], solY[math.floor(parr[i]*len(solX))]],[solX[math.floor(parr[i]*len(solX))+1], solY[math.floor(parr[i]*len(solX))+1]], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='k', zOrder=zd)
#            dirsRev = dirsRev + arrow_dyn2([solXrev[math.floor(parr[i]*len(solXrev))+1], solYrev[math.floor(parr[i]*len(solXrev))+1]], [solXrev[math.floor(parr[i]*len(solXrev))], solYrev[math.floor(parr[i]*len(solXrev))]], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='k', zOrder=zd)
#        return (psol + psolRev + dirs + dirsRev)
        return (psol + psolRev + dirs)
    elif payMtx[0].shape == (2, 2): #AS_2P2S
        x0, y0 = X0
        sol = odeint(dynamics.testrep, [x0, y0], t, (payMtx,))
        solRev = odeint(dynamics.testrepRev, [x0, y0], t, (payMtx,))
        solX=sol[:,0]
        solY=sol[:,1]
#        solZ = [0 for i in range(len(solX))]
        solXrev=solRev[:,0]
        solYrev=solRev[:,1]
        psol = plt.plot(solX,solY,color=col,zorder=zd)
        psolRev = plt.plot(solXrev,solYrev,color=col,zorder=zd)
        dirs = arrow_dyn2([solX[math.floor(parr[0]*len(solX))],solY[math.floor(parr[0]*len(solX))]], [solX[math.floor(parr[0]*len(solX))+1],solY[math.floor(parr[0]*len(solX))+1]], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color=col, zOrder=zd)
#        dirsRev = arrow_dyn2([solXrev[math.floor(parr[0]*len(solXrev))],solYrev[math.floor(parr[0]*len(solXrev))]], [solXrev[math.floor(parr[0]*len(solXrev))+1],solYrev[math.floor(parr[0]*len(solXrev))+1]], fig, ax, arrow_width=arrWidth, arrow_size=arrSize,arrow_color=col,zOrder=zd)
        for i in range(1, len(parr)):
            dirs = dirs+arrow_dyn2([solX[math.floor(parr[i]*len(solX))],solY[math.floor(parr[i]*len(solX))]], [solX[math.floor(parr[i]*len(solX))+1],solY[math.floor(parr[i]*len(solX))+1]], fig, ax, arrow_width=arrWidth, arrow_size=arrSize,arrow_color=col,zOrder=zd)
#            dirsRev = dirsRev + arrow_dyn2([solXrev[math.floor(parr[i]*len(solXrev))+1],solYrev[math.floor(parr[i]*len(solXrev))+1]], [solXrev[math.floor(parr[i]*len(solXrev))],solYrev[math.floor(parr[i]*len(solXrev))]], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color=col, zOrder=zd)
    elif payMtx[0].shape == (4,):
        x0, y0, z0 = X0
        sol = odeint(dynamics.repDyn4, [x0, y0, z0], t, (payMtx,))
        solRev = odeint(dynamics.repDyn4Rev, [x0, y0, z0], t, (payMtx,))
        solX, solY, solZ = [], [], []
        solXrev, solYrev, solZrev = [], [], []
        for pt in sol:
            cPt = eqsol.sim_to_p_2P4S(pt[0], pt[1], pt[2])
            solX.append(cPt[0])
            solY.append(cPt[1])
            solZ.append(cPt[2])
        for pt in solRev:
            cPt = eqsol.sim_to_p_2P4S(pt[0],pt[1], pt[2])
            solXrev.append(cPt[0])
            solYrev.append(cPt[1])
            solZrev.append(cPt[2])
#        nb_lines = 50
#        lines_step = int(nb_increments/nb_lines)
#        for i in range(0, nb_increments - 1, lines_step):
#            colz = plt.cm.jet_r((i/nb_increments))
#            psol = ax.plot(solX[i:i+lines_step], solY[i:i+lines_step], solZ[i:i+lines_step], linewidth = 1.5, color=colz, zorder=zd)
            #psolRev = ax.plot(solXrev[i:i+2], solYrev[i:i+2], solZrev[i:i+2], color=colz, zorder=zd)
        psol = ax.plot(solX, solY, solZ, linewidth = 0.8, color=col, zorder=zd)
        psolRev = ax.plot(solXrev, solYrev, solZrev, linewidth = 0.8, color='orange', zorder=zd)
        dirs = arrow_dyn3([solX[math.floor(parr[0]*len(solX))], solY[math.floor(parr[0]*len(solX))], solZ[math.floor(parr[0]*len(solX))]], [solX[math.floor(parr[0]*len(solX))+1], solY[math.floor(parr[0]*len(solX))+1], solZ[math.floor(parr[0]*len(solX))+1]], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='k', zOrder=zd)
        
#        dirsRev = arrow_dyn3([solXrev[math.floor(parr[0]*len(solXrev))], solYrev[math.floor(parr[0]*len(solXrev))], solZrev[math.floor(parr[0]*len(solXrev))]],[solXrev[math.floor(parr[0]*len(solXrev))+1], solYrev[math.floor(parr[0]*len(solXrev))+1], solZrev[math.floor(parr[0]*len(solXrev))+1]],fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='g', zOrder=zd)
        for i in range(1, len(parr)-1):
            dirs = dirs + arrow_dyn3([solX[math.floor(parr[i]*len(solX))], solY[math.floor(parr[i]*len(solX))], solZ[math.floor(parr[i]*len(solX))]],[solX[math.floor(parr[i]*len(solX))+1], solY[math.floor(parr[i]*len(solX))+1], solZ[math.floor(parr[i]*len(solX))+1]], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='r', zOrder=zd)
        dirs = dirs + arrow_dyn3([solX[math.floor(parr[-1]*len(solX))], solY[math.floor(parr[-1]*len(solX))], solZ[math.floor(parr[-1]*len(solX))]], [solX[math.floor(parr[-1]*len(solX))+1], solY[math.floor(parr[-1]*len(solX))+1], solZ[math.floor(parr[-1]*len(solX))+1]], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='k', zOrder=zd)

#            dirsRev = dirsRev + arrow_dyn3([solXrev[math.floor(parr[i]*len(solXrev))+1], solYrev[math.floor(parr[i]*len(solXrev))+1], 0], [solXrev[math.floor(parr[i]*len(solXrev))], solYrev[math.floor(parr[i]*len(solXrev))], 0], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='g', zOrder=zd)
#        return (psol + psolRev + dirs + dirsRev)
        #return(psol+psolRev+dirs)
    return None

def equilibria(payMtx, ax, colSnk, colSdl, colSce, ptSize, zd):
    """Computes the equilibrium points of the game and characterizes them."""
    source = [] # list of sources (both eigenvalues are positive)
    sink = []   # list of sinks (both eigenvalues are negative)
    saddle = [] # list of saddles (one neg., one pos. eig.)
    centre = [] # list of centres (pure complex eigenvalues)
    undet = []  # list of equilibria with both 0 eigenvalues
    numEqs = []
    numEig = []
    
    nuEqsRaw = eqsol.solGame(payMtx) #Computes equilibria of the replicator dynamics
    if payMtx[0].shape == (3,):
        for i in range(len(nuEqsRaw)): # Checks that all equilibria are within the simplex
            if (0 <=  nuEqsRaw[i][0] <= 1 and 0 <=  nuEqsRaw[i][1] <= 1 and nuEqsRaw[i][0] + nuEqsRaw[i][1] <= 1):
                numEqs += [[nuEqsRaw[i][0],nuEqsRaw[i][1]]]
        for i in range(len(numEqs)): # Checks that equilibria are real
            if (numEqs[i][0].imag !=0 or numEqs[i][1].imag != 0):
                numEqs[i][0] = 99 # Attributes unrealistic values in that case
                numEqs[i][1] = 99
        for i in range(len(numEqs)): # Computes eigenvalues of Jacobian evaluated at each equilibrium
            t = 0
            X = Matrix(dynamics.repDyn3([x,y], t, payMtx))
            Y = Matrix([x, y])
            JC = X.jacobian(Y)
            valuedJC = np.array(JC.subs([(x, numEqs[i][0]), (y, numEqs[i][1])]))
            M = np.zeros(valuedJC.shape)
            for i in range(len(valuedJC)):
                for j in range(len(valuedJC)):
                    M[i][j] = valuedJC[i][j]
            w, v = np.linalg.eig(M)
            numEig.append(w)
                
    elif payMtx[0].shape == (2, 2): # Same thing but for other type of game
        for i in range(len(nuEqsRaw)):
            if (0 <=  nuEqsRaw[i][0] <= 1 and 0 <=  nuEqsRaw[i][1] <= 1):
                numEqs += [[nuEqsRaw[i][0],nuEqsRaw[i][1]]]
        for i in range(len(numEqs)): 
            if (numEqs[i][0].imag !=0 or numEqs[i][1].imag != 0):
                numEqs[i][0] = 99
                numEqs[i][1] = 99
        for i in range(len(numEqs)):
            t = 0
            X = Matrix(dynamics.testrep([x, y], t, payMtx))
            Y = Matrix([x, y])
            JC = X.jacobian(Y)
            valuedJC = np.array(JC.subs([(x, numEqs[i][0]), (y, numEqs[i][1])]))
            M = np.zeros(valuedJC.shape)
            for i in range(len(valuedJC)):
                for j in range(len(valuedJC)):
                    M[i][j] = valuedJC[i][j]
            w, v = np.linalg.eig(M)
            numEig.append(w)
    
    elif payMtx[0].shape == (4,):
        for i in range(len(nuEqsRaw)): # Checks that all equilibria are within the simplex
            if (0 <=  nuEqsRaw[i][0] <= 1 and 0 <=  nuEqsRaw[i][1] <= 1 and 0 <=  nuEqsRaw[i][2] <= 1 and nuEqsRaw[i][0] + nuEqsRaw[i][1] + nuEqsRaw[i][2]<= 1):
                numEqs += [[nuEqsRaw[i][0], nuEqsRaw[i][1], nuEqsRaw[i][2]]]
        for i in range(len(numEqs)): # Checks that equilibria are real
            if (numEqs[i][0].imag !=0 or numEqs[i][1].imag != 0 or numEqs[i][2].imag != 0):
                numEqs[i][0] = 99 # Attributes unrealistic values in that case
                numEqs[i][1] = 99
                numEqs[i][2] = 99
        for i in range(len(numEqs)): # Computes eigenvalues of Jacobian evaluated at each equilibrium
            t = 0
            X = Matrix(dynamics.repDyn4([x, y, z], t, payMtx))
            Y = Matrix([x, y, z])
            JC = X.jacobian(Y)
            valuedJC = np.array(JC.subs([(x, numEqs[i][0]), (y, numEqs[i][1]), (z, numEqs[i][2])]))
            M = np.zeros(valuedJC.shape)
            for i in range(len(valuedJC)):
                for j in range(len(valuedJC)):
                    M[i][j] = valuedJC[i][j]
            w, v = np.linalg.eig(M)
            numEig.append(w)
        
    for i in range(len(numEqs)): # Classify equilibria
        numEqs[i] = [round(num, 10) for num in numEqs[i]]
        numEig[i] = [round(num, 12) for num in numEig[i]]
        if payMtx[0].shape == (2, 2): point_to_plot = np.array([numEqs[i][0], numEqs[i][1]])
        elif payMtx[0].shape == (3,): 
            point_to_plot = np.array(eqsol.p_to_sim(numEqs[i][0] , numEqs[i][1]))
            numEqs[i].append(1 - numEqs[i][0] - numEqs[i][1])
        elif payMtx[0].shape == (4,): point_to_plot = np.array(eqsol.sim_to_p_2P4S(numEqs[i][0] , numEqs[i][1], numEqs[i][2]))
        print("FP", [round(num, 2) for num in numEqs[i]], "| eigVs", [round(num, 2) for num in numEig[i]])
        if (0<=numEqs[i][0]<=1 and 0<=numEqs[i][1]<=1):
            l1, l2 = numEig[i][0], numEig[i][1]
            suml, prodl = l1+l2, l1*l2
            if prodl<0: saddle.append(point_to_plot)
            else:
                if suml>0: source.append(point_to_plot)
                elif suml<0: sink.append(point_to_plot)
                else:
                    if l1.imag !=0:
                        centre.append(point_to_plot)
                    else: undet.append(point_to_plot)
                
    #Plot equilibria
    saddlexs, saddleys, saddlezs = [], [], []
    sinkxs, sinkys, sinkzs = [], [], []
    sourcexs, sourceys, sourcezs = [], [], []
    centrexs, centreys, centrezs = [], [], []
    undetxs, undetys, undetzs = [], [], []
    for pt in source:
        sourcexs.append(pt[0])
        sourceys.append(pt[1])
        if payMtx[0].shape == (4,):
            sourcezs.append(pt[2])
    for pt in saddle:
        saddlexs.append(pt[0])
        saddleys.append(pt[1])
        if payMtx[0].shape == (4,):
            sourcezs.append(pt[2])
    for pt in centre :
        centrexs.append(pt[0])
        centreys.append(pt[1])
        if payMtx[0].shape == (4,):
            sourcezs.append(pt[2])
    for pt in sink:
        sinkxs.append(pt[0])
        sinkys.append(pt[1])
        if payMtx[0].shape == (4,):
            sourcezs.append(pt[2])
    for pt in undet:
        undetxs.append(pt[0])
        undetys.append(pt[1])
        if payMtx[0].shape == (4,):
            sourcezs.append(pt[2])
#    ax.scatter(sinkxs, sinkys, sinkzs, s=ptSize, color=colSnk, marker='o', edgecolors='black', alpha=1, depthshade=False, zorder=zd)
#    ax.scatter(sourcexs, sourceys, sourcezs, s=ptSize, color=colSce, marker='o', edgecolors='black', alpha=1, depthshade=False, zorder=zd)
#    ax.scatter(saddlexs, saddleys, saddlezs, s=ptSize, color=colSdl, marker='o', edgecolors='black', alpha=1, depthshade=False, zorder=zd)
#    ax.scatter(centrexs, centreys, centrezs, s=ptSize, color='orange', marker='*', edgecolors='black', alpha=1, depthshade=False, zorder=zd)
#    ax.scatter(undetxs, undetys, undetzs, s=ptSize, color='red', marker='x', edgecolors='black', alpha=1, depthshade=False, zorder=zd)
    ax.scatter(sinkxs, sinkys, s=ptSize, color=colSnk, marker='o', edgecolors='black', alpha=1, zorder=zd)
    ax.scatter(sourcexs, sourceys, s=ptSize, color=colSce, marker='o', edgecolors='black', alpha=1, zorder=zd)
    ax.scatter(saddlexs, saddleys, s=ptSize, color=colSdl, marker='o', edgecolors='black', alpha=1, zorder=zd)
    ax.scatter(centrexs, centreys, s=ptSize, color='orange', marker='*', edgecolors='black', alpha=1, zorder=zd)
    ax.scatter(undetxs, undetys, s=ptSize, color='red', marker='x', edgecolors='black', alpha=1, zorder=zd)
    if payMtx[0].shape == (4,):
        ax.scatter(sinkxs, sinkys, sinkzs, s=ptSize, color=colSnk, marker='o', edgecolors='black', alpha=1, zorder=zd)
        ax.scatter(sourcexs, sourceys, sourcezs, s=ptSize, color=colSce, marker='o', edgecolors='black', alpha=1, zorder=zd)
        ax.scatter(saddlexs, saddleys, saddlezs, s=ptSize, color=colSdl, marker='o', edgecolors='black', alpha=1, zorder=zd)
        ax.scatter(centrexs, centreys, centrezs, s=ptSize, color='orange', marker='*', edgecolors='black', alpha=1, zorder=zd)
        ax.scatter(undetxs, undetys, undetzs, s=ptSize, color='red', marker='x', edgecolors='black', alpha=1, zorder=zd)
#    return [pSink] + [pSource] + [pSaddle] + [pUndet]
    return [source] + [saddle] + [sink] + [centre] + [undet]
        

def matrix_to_colors(matrix, cmap):
    """Converts a matrix into a RGBA color map.""" 
    color_dimension = matrix # It must be in 2D - as for "X, Y, Z".
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)
    return fcolors, m

def speed_plot(x_region, y_region, step, payMtx, ax, cmap, levels, zorder):
    """Plots game dynamics (speed of movement in the simplex)"""
    x = np.linspace(x_region[0], x_region[1], step)
    y = np.linspace(y_region[0], y_region[1], step)
    X, Y = np.meshgrid(x, y)
    X, Y = eqsol.outofbounds_reproject(X, Y)
    C = eqsol.speedGrid(X, Y, payMtx)
    surf = ax.contourf(X, Y, C, levels=levels, cmap=cmap, corner_mask = False, alpha=0.9)
    return surf