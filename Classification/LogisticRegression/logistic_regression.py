import numpy as np
import matplotlib.pyplot as plt
import pandas



data = pandas.read_csv("../../data/Default.csv")

default = [0 if d=='No' else 1 for d in data['default']]
default_colors = ["dodgerblue" if d=='No' else "firebrick" for d in data['default']]


def prob(b0, b1, x):
    return np.exp(b0+b1*x) / (1 + np.exp(b0+b1*x))

def lnprob(b0, b1, x):
    return (b0+b1*x) - np.log(1 + np.exp(b0+b1*x))




def lnlikelihood(b0, b1, data):
    total = 0
    for ii in range(len(data)):
        thisln = 0
        if data['default'][ii] == "Yes":
            #thisln = np.log(prob(b0, b1, data['balance'][ii]))
            thisln = lnprob(b0, b1, data['balance'][ii])
        if data['default'][ii] == "No":
            thisln = np.log(1-prob(b0, b1, data['balance'][ii]))
        total += thisln 

    return total

b0s = np.linspace(-32, 5, 300)
b1s = np.linspace(-0.03, 0.02, 300)

B0, B1 = np.meshgrid(b0s, b1s)

res = lnlikelihood(B0, B1, data)
ind = np.argmax(res)
best_b0 = B0.flatten()[ind]
best_b1 = B1.flatten()[ind]

fig, ax = plt.subplots()
ax.scatter(data['balance'], default, color=default_colors)

balance_arr = np.linspace(0, 3000, 1000)
ax.plot(balance_arr, prob(best_b0, best_b1, balance_arr), color='black')


ax.set_xlabel('Balance')
ax.set_ylabel("Default")
ax.set_yticks([0,1])
ax.set_yticklabels(["No", "Yes"])
#fig, ax = plt.subplots()
#ax.contour(B0, B1, res, 300)
#fig, ax = plt.subplots()
#ax.imshow(res, extent=(np.min(b0s), np.max(b0s), np.min(b1s), np.max(b1s)), aspect='auto', origin='lower')
plt.show()
