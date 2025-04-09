#%% 生成モデル
#%%
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

# %%
def normal(x, mu=0, sigma=1):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

# %%
x = np.linspace(-5, 5, 100)
y = normal(x)

plt.plot(x,y)

# %%
N = 3

xs = []

for n in range(N):
    x = np.random.rand()
    xs.append(x)

x_mean = np.mean(xs)
print(x_mean)
# %%

x_mans = []
N = 10

for _ in range(10000):
    xs = []
    for n in range(N):
        x = np.random.rand()
        xs.append(x)
    x_mean = np.mean(xs)
    x_mans.append(x_mean)

plt.hist(x_mans, bins=100, density=True)
plt.title(f'N={N}')
plt.xlabel('x')
plt.ylabel('density')
plt.xlim(-0.05, 1.05)
plt.ylim(0, 5)
plt.show()

# %%
x_sums = []
N = 3

for _ in range(10000):
    xs = []
    for n in range(N):
        x = np.random.rand()
        xs.append(x)
    t = np.sum(xs)
    x_sums.append(t)

def normal(x, mu=0, sigma=1):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

x_norm = np.linspace(-5, 5, 1000)
mu = N / 2
sigma = np.sqrt(N/12)
y_norm = normal(x_norm, mu, sigma)

plt.hist(x_sums, bins=100, density=True)
plt.plot(x_norm, y_norm)
plt.title(f'N={N}')
plt.xlabel('x')

# %%
