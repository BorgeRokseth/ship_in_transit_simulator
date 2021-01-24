import numpy as np
from scipy.stats import entropy
from math import log, e
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import timeit
import sys
import math

def entropy1(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)

def entropy2(labels, base=None):
  """ Computes entropy of label distribution. """

  n_labels = len(labels)

  if n_labels <= 1:
    return 0

  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)

  return ent

def entropy3(labels, base=None):
  vc = pd.Series(labels).value_counts(normalize=True, sort=False)
  base = e if base is None else base
  return -(vc * np.log(vc)/np.log(base)).sum()

def entropy4(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  norm_counts = counts / counts.sum()
  base = e if base is None else base
  return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

labels = [1,3,5,2,3,5,3,2,1,3,4,5]

#print(entropy1(labels))
#print(entropy2(labels))
#print(entropy3(labels))
#print(entropy4(labels))
DISTRIBUTIONS = [
  st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2, st.cosine,
  st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife, st.fisk,
  st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
  st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
  st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma, st.invgauss,
  st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace, st.levy, st.levy_l, st.levy_stable,
  st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2, st.ncf,
  st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
  st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm, st.tukeylambda,
  st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
]
rv = st.norm(100, 2)
c1 = 1.79
c2 = 3
x1 = np.linspace(st.weibull_min.ppf(0.01, c1), st.weibull_min.ppf(0.99, c1), 100)
x2 = np.linspace(st.weibull_min.ppf(0.01, c2), st.weibull_min.ppf(0.99, c2), 100)
print(st.weibull_min.mean(c1, loc=0, scale=1))
print(st.weibull_min.var(c1, loc=0, scale=1))

print(st.weibull_min.entropy(c1, loc=0, scale=1))
print(st.weibull_min.mean(c2, loc=0, scale=1))
print(st.weibull_min.var(c2, loc=0, scale=1))
print(st.weibull_min.entropy(c2, loc=0, scale=1))
plt.plot(x1, st.weibull_min.pdf(x1, c1))
plt.plot(x2, st.weibull_min.pdf(x2, c2))
plt.show()
print(rv.entropy())

P=[]
for i in range(10):
  pos1 = (36*i, 72*i)
  pos2 = (2400, 0)
  r1 = 2500+i*50
  r2 = 500
  def intersecting_area(pos1, pos2, r1, r2):
    """pos1, pos2 are the positions of circle 1 and 2, r1, r2 are the radius."""
    d = (((pos1[0]-pos2[0])**2)+((pos1[1]-pos2[1])**2))**0.5
    small_circle_area = math.pi*(min(r1, r2)**2)
    if d <= (max(r1, r2)-min(r1, r2)):
      area = small_circle_area
      print('Intersecting area will be area of smaller circle')

    elif d >= (r1+r2):
      area = 0
      print('No intersecting area')
    else:
      angle1 = ((r1 * r1) + (d * d) - (r2 * r2)) / (2 * r1 * d)
      angle2 = ((r2 * r2) + (d * d) - (r1 * r1)) / (2 * r2 * d)
      theta1 = (math.acos(angle1) * 2)
      theta2 = (math.acos(angle2) * 2)
      area1 = (0.5 * theta1 * (r1 * r1)) - (0.5 * d * r2 * math.sin(theta2))
      area2 = (0.5 * theta2 * (r2 * r2)) - (0.5 * d * r1 * math.sin(theta1))
      area = area1 + area2
    return area

  p = intersecting_area(pos1, pos2, r1, r2)/(math.pi*(r1**2))
  P.append(p)
  print(i)
total_p = 1-(1-P[0])*(1-P[1])*(1-P[2])*(1-P[3])*(1-P[4])*(1-P[5])*(1-P[6])*(1-P[7])*(1-P[8])*(1-P[9])
print(p)
print(P)
print(total_p)
print(math.pi*(r2**2))
print(intersecting_area(pos1, pos2, r1, r2))
print(math.pi*(r1**2))




