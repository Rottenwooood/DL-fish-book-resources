import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,6,0.1)# begin,end,pace
y1 = np.sin(x)
y2 = np.cos(x)

# draw gragh
plt.plot(x,y1,label="sin") # sin
plt.plot(x,y2,linestyle="--",label="cos") # cos
plt.xlabel("x") # x
plt.ylabel("y") # y 
plt.title('sin & cos') # title
plt.legend()
plt.show()
