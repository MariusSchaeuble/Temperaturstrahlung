import matplotlib
import numpy
import numpy as np
import sympy as sym
from Helpers import identifier, isCharacter
import math
from numpy import matrix, array, mean, std, max, linspace, ones, sin, cos, tan, arctan, pi, sqrt, exp, arcsin, arccos, arctan2, sinh, cosh, zeros, log
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, xlabel, ylabel, legend, title, savefig, errorbar, grid
import scipy.optimize as opt
from GPII import *
from math import sqrt
pi = math.pi


matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)







def gauss(term):
    ids = identifier(term)
    symbols = []
    for str1 in ids:
        symbols.append(sym.sympify(str1))
    termSymbol = sym.sympify(term)
    values = []
    for ident in ids:
        exec("values.append(" + ident + ")")

    derivatives = []
    i = 0
    while i < len(symbols):
        r = sym.diff(termSymbol, symbols[i])
        j = 0
        while j < len(symbols):
            # exec('r.evalf(subs={symbols[j]: ' + values[j] + '})')
            r = r.evalf(subs={symbols[j]: values[j]})
            j += 1
        derivatives.append(r.evalf())
        i += 1
    i = 0
    while i < len(derivatives):
        exec("derivatives[i] *= sigma_" + ids[i])
        i = i + 1
    res = 0
    for z in derivatives:
        res += z ** 2
    return math.sqrt(res)

def gaussVec(term):
    ids = identifier(term)
    arrays = []
    for i in range(len(ids)):
        if isinstance(eval(ids[i]), np.ndarray):
            arrays.append(ids[i])
    arrayLength = len(eval(arrays[0]))
    symbols = []
    for str1 in ids:
        symbols.append(sym.sympify(str1))
    termSymbol = sym.sympify(term)
    res = []
    for k in range(arrayLength):
        values = []
        for ident in ids:
            if ident in arrays:
                exec("values.append(" + ident + "[k]" + ")")
            else:
                exec("values.append(" + ident + ")")
        derivatives = []
        i = 0
        while i < len(symbols):
            r = sym.diff(termSymbol, symbols[i])
            j = 0
            while j < len(symbols):
                r = r.evalf(subs={symbols[j]: values[j]})
                j += 1
            derivatives.append(r.evalf())
            i += 1
        i = 0
        sigmaArrays = []
        for t in range(len(ids)):
            if isinstance(eval("sigma_" + ids[t]), np.ndarray):
                sigmaArrays.append(ids[t])
        while i < len(derivatives):
            if ids[i] in sigmaArrays:
                exec("derivatives[i] *= sigma_" + ids[i] + "[k]")
            else:
                exec("derivatives[i] *= sigma_" + ids[i])
            i = i + 1
        resj = 0
        for z in derivatives:
            resj += z ** 2
        res.append(sqrt(resj))
    return array(res)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def linear(x, a, b):
    return a*x + b


# Duchmesser aufnehmen

rb = 2/100
rs = 2/100
sigma_rb = 0.5/1000
sigma_rs = 0.5/1000

A = pi*rb**2
sigma_A = gauss("pi*rb**2")
#messschieber, annahme 0.5mm fhler

#Raumtemp am Anfang

Ti = 24.8 + 273
Tf = 26.3 + 273

#leslie
r = 35/100
#lineal

T1 = matrix("""
83.6
82.5
81.2
81.2
""")

U1 = matrix("""
0.365
0.124
0.348
0.133
""")#schwarz, glänzend, weis, matt


T2 = matrix("""
79.8
78.3
78.1
78.2
""")

U2 = matrix("""
0.351
0.12
0.337
0.136
""")#schwarz, glänzend, weis, matt



#r^2 abhängigkeit

T_ofen_anfang = 348.3 + 273.15
T_ofen_ende = 346.9 + 273.15

W21 = matrix("""
0.872    25;
0.626    30;
0.485    35;
0.389    40;
0.323    45;
0.275    50
""")# messung mit dem Lineal der achse: U in V, r in cm

W22 = matrix("""
0.86     25;
0.611    30;
0.461    35;
0.375    40;
0.311    45;
0.265    50
""")# messung mit dem Lineal der achse: U in V, r in cm


U1 = toArray(W21[:, 0])
sigma_U1 = 0.2*U1

U2 = toArray(W22[:, 0])
sigma_U2 = 0.2*U2

U = (U1 + U2)/2
sigma_U = gaussVec("(U1 + U2)/2")

r1 = toArray(W21[:, 1])/100
sigma_r1 = 1/100

r2 = toArray(W22[:, 1])/100
sigma_r2 = 0.2*r2

Ulog = log(U)
sigma_Ulog = gaussVec("log(U)")

rlog = log(r1)
sigma_rlog = gaussVec("log(r1)")# nur log muss bestimmt werden

errorbar(rlog, Ulog, sigma_Ulog, sigma_rlog,'x', label='Spannungsmessung')
optimizedParameters1, s = opt.curve_fit(linear, rlog, Ulog)
plot(rlog, linear(rlog, *optimizedParameters1), label="fit")
xlabel('log(r/m)', fontsize=20)
ylabel('log(U/V)', fontsize=20)
legend(fontsize=13, loc='upper right')
grid()
plt.tight_layout()
savefig('rq')
show()

rexp = optimizedParameters1[0]
sigma_rexp = np.diag(s)[0]



#absenkung der temp.
W3 = matrix("""
0.86    346.9;
0.784   336.9;
0.724   326.9;
0.673   316.9;
0.633   306.9;
0.593   296.9;
0.550   286.9;
0.507   273.4;
0.479   266.9;
0.450   256.9;
0.418   246.9;
0.386   236.9;
0.366   226.9;
0.336   216.9 ;
0.311   206.9;
0.287   196.9;
0.267   186.9;
0.245   176.9;
0.228   166.9;
0.211   156.9;
0.195   146.9
""")# abstand 25cm: U in V, T in °C

r = 25/100
sigma_r = 1/100


T = toArray(W3[:, 1])[::-1]
sigma_T = 0.2*T
sigma_Tf = 0.1
Y = ((T + 273.15)**4 - Tf**4)*10**-9
sigma_Y = gaussVec("(T + 273.15)**4 - Tf**4")*10**-9

U = toArray(W3[:, 0])[::-1]
sigma_U = 0.2*U


errorbar(U, Y, sigma_Y, sigma_U,'x', label='Temperaturmessung')
optimizedParameters1, s = opt.curve_fit(linear, U, Y)
plot(U, linear(U, *optimizedParameters1), label="fit")
xlabel('U', fontsize=20)
ylabel('T^4 - T0^4', fontsize=20)
legend(fontsize=13, loc='upper left')
grid()
plt.tight_layout()
savefig('boltz')
show()

steig = optimizedParameters1[0]*10**9
sigma_steig = np.diag(s)[0]*10**9

sigma = (pi*r**2)/(0.16*1000*A**2*steig)


#glühbirne

W4 = matrix("""
1.6    1.57    809 ;
1.9    1.72    942 ;
2.1    1.82    1020;
2.3    1.99    1133;
2.5    2.07    1195;
2.9    2.23    1304;
3.3    2.44    1405;
3.9    2.62    1514;
4.3    2.81    1607;
5.3    3.15    1758;
5.9    3.33    1838;
6.7    3.59    1939;
7.5    3.81    2034;
8.4    4.08    2129;
9.5    4.38    2233;
10.5   4.65    2332
""")# U, I, T in °C
#leistung 20% fehler

#P = W4[:, 0]*W4[:, 1]
P = np.multiply(W4[:, 0], W4[:, 1])
P = toArray(P)
sigma_P = 0.2*P

T_birne = toArray(W4[:, 2])
sigma_T_birne = zeros(len(T_birne))
for i in range(0, len(T_birne)):
    if T_birne[i] < 1500:
        sigma_T_birne[i] = max([4, 0.005*T_birne[i]])
    else:
        sigma_T_birne[i] = 0.0075*T_birne[i]


T_birne = T_birne + 273.15
#fehlerrechnung in anleitung angegeben für 23°C

errorbar(T_birne, P, sigma_P, sigma_T_birne,'x', label='Temperaturmessung')
xlabel('T in K', fontsize=20)
ylabel('P in W', fontsize=20)
legend(fontsize=13, loc='center left')
grid()
plt.tight_layout()
savefig('birne')
show()

T_birne_log = log(T_birne)
sigma_T_birne_log = gaussVec("log(T_birne)")

P_log = log(P)
sigma_P_log = gaussVec("log(P)")



errorbar(T_birne_log, P_log, sigma_P_log, sigma_T_birne_log,'x', label='Temp., logarithmisch')
optimizedParameters1, s = opt.curve_fit(linear, T_birne_log, P_log)
plot(T_birne_log, linear(T_birne_log, *optimizedParameters1), label="fit")
xlabel('log(T/K)', fontsize=20)
ylabel('log(P/W)', fontsize=20)
legend(fontsize=13, loc='upper left')
grid()
plt.tight_layout()
savefig('birnelog')
show()




