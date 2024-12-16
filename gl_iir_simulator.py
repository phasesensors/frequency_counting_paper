import math

import numpy as np
import glotlib


# Initial conditions.
a = 2**-16
y = 0
z = 0
p = 0
q = 0
r = 0
Y = []
Z = []
P = []
Q = []
R = []

PREFIX_ZEROES = 100000

# Compute the response.
X = [0]*PREFIX_ZEROES + [1]*1000000
# X = ([1] + [0]*4999) * 10
# for i, x in enumerate(X):
for i, x in enumerate(X):
    y = a*x + (1-a)*y
    z = a*y + (1-a)*z
    p = a*z + (1-a)*p
    q = a*p + (1-a)*q
    r = a*q + (1-a)*r

    # y = (1-a)**i*a
    # z = (i+1)*(1-a)**i*a**2
    # p = ((i+1)*(i+2)/2)*(1-a)**i*a**3
    # q = ((i+1)*(i+2)*(i+3)/6)*(1-a)**i*a**4
    # r = ((i+1)*(i+2)*(i+3)*(i+4)/24)*(1-a)**i*a**5
    Y.append(y)
    Z.append(z)
    P.append(p)
    Q.append(q)
    R.append(r)

# Find Tau for each filter.
percent_99 = 1 - (1 / math.e)**5
for ty, y in enumerate(Y):
    if y >= percent_99:
        ty -= PREFIX_ZEROES
        print('ty = %u' % ty)
        break
for tz, z in enumerate(Z):
    if z >= percent_99:
        tz -= PREFIX_ZEROES
        print('tz = %u' % tz)
        break
for tp, p in enumerate(P):
    if p >= percent_99:
        tp -= PREFIX_ZEROES
        print('tp = %u' % tp)
        break
for tq, q in enumerate(Q):
    if q >= percent_99:
        tq -= PREFIX_ZEROES
        print('tq = %u' % tq)
        break
for tr, r in enumerate(R):
    if r >= percent_99:
        tr -= PREFIX_ZEROES
        print('tr = %u' % tr)
        break

# Compute the derivative of Z.
D1 = []
D2 = []
D3 = []
for i in range(len(X)):
    D1.append(a**1*(1-a)**i*math.log(1-a))
    D2.append(a**2*(1-a)**i*((i+1)*math.log(1-a) + 1))
    D3.append(a**3*(1-a)**i*((i+1)*(i+2)*math.log(1-a) + 2*i + 3)/2)

w  = glotlib.Window(2000, 1500, msaa=4)
font = glotlib.fonts.vera(36, 2)
p  = w.add_plot(111, label_font=font, border_width=4)
# p2 = w.add_plot(312, sharex=p, label_font=font, border_width=4)
# p3 = w.add_plot(313, sharex=p, label_font=font, border_width=4)
# p.set_y_label('IIR Period', side='right')
# p2.set_y_label('IIR Freq @ 50 kHz', side='right')
# p3.set_y_label('IIR Freq @ 40 kHz', side='right')

XX = np.arange(len(X)) - PREFIX_ZEROES
sx = p.add_steps(X=XX, Y=X, width=5)
sy = p.add_steps(X=XX, Y=Y, width=5)
sz = p.add_steps(X=XX, Y=Z, width=5)
sp = p.add_steps(X=XX, Y=P, width=5)
sq = p.add_steps(X=XX, Y=Q, width=5)
sr = p.add_steps(X=XX, Y=R, width=5)
p.add_vline(ty, color=sy.color, width=5)
p.add_vline(tz, color=sz.color, width=5)
p.add_vline(tp, color=sp.color, width=5)
p.add_vline(tq, color=sq.color, width=5)
p.add_vline(tr, color=sr.color, width=5)
p.snap_bounds()

# p2.add_steps(X=XX, Y=D1, width=1)
# p2.add_steps(X=XX, Y=D2, width=1)
# p2.add_steps(X=XX, Y=D3, width=1)
# p2.snap_bounds()

# # Compute the frequencies from the IIR outputs, assuming a 50 kHz crystal
# # being measured and a 160 MHz reference crystal.
# N = 160e6 / 50e3    # Nominal number of ref ticks per pressure tick.
# Yf = [0] * len(Y)
# Zf = [0] * len(Y)
# Pf = [0] * len(Y)
# Qf = [0] * len(Y)
# Rf = [0] * len(Y)
# for i in range(len(X)):
#     Yf[i] = 160e6 / (N + Y[i])
#     Zf[i] = 160e6 / (N + Z[i])
#     Pf[i] = 160e6 / (N + P[i])
#     Qf[i] = 160e6 / (N + Q[i])
#     Rf[i] = 160e6 / (N + R[i])
# p2.add_steps(X=[], Y=[])
# p2.add_steps(X=XX, Y=Yf, width=5)
# p2.add_steps(X=XX, Y=Zf, width=5)
# p2.add_steps(X=XX, Y=Pf, width=5)
# p2.add_steps(X=XX, Y=Qf, width=5)
# p2.add_steps(X=XX, Y=Rf, width=5)
# 
# N = 160e6 / 40e3    # Nominal number of ref ticks per pressure tick.
# for i in range(len(X)):
#     Yf[i] = 160e6 / (N + Y[i])
#     Zf[i] = 160e6 / (N + Z[i])
#     Pf[i] = 160e6 / (N + P[i])
#     Qf[i] = 160e6 / (N + Q[i])
#     Rf[i] = 160e6 / (N + R[i])
# p3.add_steps(X=[], Y=[])
# p3.add_steps(X=XX, Y=Yf, width=5)
# p3.add_steps(X=XX, Y=Zf, width=5)
# p3.add_steps(X=XX, Y=Pf, width=5)
# p3.add_steps(X=XX, Y=Qf, width=5)
# p3.add_steps(X=XX, Y=Rf, width=5)

glotlib.interact()
