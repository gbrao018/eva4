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

import math
def draw_cyclic_lr_graph1(lr_min,lr_max,batch_size,max_num_cycles,no_of_images):
  #lr_min,lr_max are in y direction
  # each triangle to be plot with 3 points (0,lr_min),(0+step,lr_max),(2*step,lr_max) 
  x_points = []
  y_points = []
  plt.figure(figsize=(20,10))
  cycle_count =0
  iteration =0
  batch_iterations = no_of_images/batch_size
  step_size = 10 *batch_iterations # between 2 to 10 iterations

  while(cycle_count <= max_num_cycles):
    cycle_count = math.floor(1+iteration/(2*step_size))
    x = abs((iteration/step_size) - (2*(cycle_count))+1)
    lr_t = lr_min + ((lr_max - lr_min)*(1.0 - x))
    y_points.append(lr_t)
    x_points.append(iteration)
    iteration = iteration +1

  plt.plot(x_points, y_points, '-')
    #print(x_points[i],'::',y_points[i])
  
  plt.xlabel('Iterations')
  plt.ylabel('Lr Range')
  plt.show()
