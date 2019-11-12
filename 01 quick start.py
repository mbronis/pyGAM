
# https://pygam.readthedocs.io/en/latest/notebooks/quick_start.html

from pygam.datasets import wage

X, y = wage()

X.shape, y.shape



from pygam import LinearGAM, s, f

gam = LinearGAM(s(0) + s(1) + f(2)).fit(X, y)
gam.summary()


#by deafaule s has 20 base functions
gam2 = LinearGAM(s(0, n_splines=5) + s(1) + f(2)).fit(X, y)
gam2.summary()


#by default all terms has lambda penalty of 0.6,
#running grid search for lambda optimization
#using GCV (generalized cv-score)

import numpy as np

lam = np.logspace(-3, 5, 5)
lams = [lam] * 3
lams

gam3 = LinearGAM(s(0) + s(1) + f(2))
gam3.gridsearch(X, y, lam=lams)
gam3.summary()



# plot pdp
import matplotlib.pyplot as plt

for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue

    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
    plt.show()


for i, term in enumerate(gam2.terms):
    if term.isintercept:
        continue

    XX = gam2.generate_X_grid(term=i)
    pdep, confi = gam2.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
    plt.show()


for i, term in enumerate(gam3.terms):
    if term.isintercept:
        continue

    XX = gam3.generate_X_grid(term=i)
    pdep, confi = gam3.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
    plt.show()




