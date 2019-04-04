import numpy as np
import matplotlib.pyplot as plt
def f(x):
 return x*x-4*x+4
def f1(x):
 return 2*x-4
def Grandient_Descent(start_x,learning_rate,f):
  x=start_x
  for i in range(20):
    grad = f1(x)
    x -= grad*learning_rate
    print(i,grad,x)
    if abs(grad) < 1e-10:
     break
  return x
Grandient_Descent(5,0.1,f)
x = np.linspace(-10,13,1000)
y = f(x)
plt.plot(x,y)
plt.show()