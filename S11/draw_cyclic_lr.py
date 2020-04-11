import numpy as np
import matplotlib.pyplot as plt

def draw_cyclic_lr_graph(lr_min,lr_max,step,iterations):
  #lr_min,lr_max are in y direction
  # each triangle to be plot with 3 points (0,lr_min),(0+step,lr_max),(2*step,lr_max) 
  x_points = [0]
  y_points = [lr_min]
  last_step=step
  max = True
  plt.figure(figsize=(20,10))
  
  for x in range(iterations):
    x_points.append(last_step)
    last_step= last_step + step
    if(max == True):
      y_points.append(lr_max)
      max = False
    else:
      y_points.append(lr_min)
      max = True
  #for i in range(iterations-1):
    plt.plot(x_points, y_points, 'bo-')
    #print(x_points[i],'::',y_points[i])
  
  plt.xlabel('Iterations')
  plt.ylabel('Lr Range')
  plt.show()
