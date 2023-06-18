import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)

def predict(a, xt):
  return a[0]+a[1]*xt

def MSE(a, x, y):
  total = 0
  for i in range(len(x)):
    total += (y[i]-predict(a,x[i]))**2
  return total

def loss(p):
  return MSE(p, x, y)

def optimize():
  p = [0,1]
  dh = 0.001
  p1 = p.copy()
  p2 = p.copy()
  p3 = p.copy()
  p4 = p.copy()
  while True:
    p1[0] += dh
    p2[0] -= dh
    p3[1] += dh
    p4[1] -= dh
    if loss(p1) < loss(p):
      p[0] = p1[0]
    elif loss(p2) < loss(p):
      p[0] = p2[0]
    elif loss(p3) < loss(p):
      p[1] = p3[1]
    elif loss(p4) < loss(p):
      p[1] = p4[1]
    else:
      break
    p1 = p.copy()
    p2 = p.copy()
    p3 = p.copy()
    p4 = p.copy()
  return p

p = optimize()

# Plot the graph
y_predicted = list(map(lambda t: p[0]+p[1]*t, x))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()