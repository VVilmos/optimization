import random
import math
import matplotlib.pyplot as plt
import numpy 
import pandas as pd


#creating csv file to write rows
df_x = pd.DataFrame(numpy.empty((9000,5), dtype = float))
df_x.columns = ['prev_x', 'prev_y', 'phi', 'speed', 'x']
df_y = pd.DataFrame(numpy.empty((9000,5), dtype = float))
df_y.columns = ['prev_x', 'prev_y', 'phi', 'speed', 'y']



trainx_row = numpy.array([])
trainy_row = numpy.array([])

fig, ax = plt.subplots()

x =  random.uniform(0 , 600)
y = random.uniform(0 , 600)
prev_x = x
prev_y = y 
trainx_row = numpy.append(trainx_row, x)
trainx_row = numpy.append(trainx_row, y)
trainy_row = numpy.append(trainy_row, x)
trainy_row = numpy.append(trainy_row, y)
ax.scatter(x, y, c='r', s=25, alpha=1, zorder = 10)

phi = random.uniform(0, 2*math.pi)
speed = random.randint(1, 150)
last_speed = speed
last_phi = phi
trainx_row = numpy.append(trainx_row, last_phi)
trainx_row = numpy.append(trainx_row, last_speed)
trainy_row = numpy.append(trainy_row, last_phi)
trainy_row = numpy.append(trainy_row, last_speed)

def plot(x1, y1, x2, y2, alfa):
    plt.xlim(-300, 1000)
    plt.ylim(-300, 1000)
    #ax.arrow(x1 + (x2-x1)*0.1, y1 + (y2-y1)*0.1, (x2-x1) + (x1-x2)*0.3, (y2-y1) + (y1-y2)*0.3, head_width=10, head_length=5, fc='k', ec='k', alpha = alfa, zorder=0)
    #ax.scatter(x2,y2, s=25, alpha = alfa+0.1, c='r', zorder = 8, marker = '+')



#random walk mobility model
#the user moves in the direction of the last movement with 60% probability
for i in range(9000):
        phi = random.uniform(last_phi- 0.25*math.pi, last_phi+ 0.25*math.pi)
        speed = random.randint(1, 150)
        phi = 0.4*phi + 0.6*last_phi
        speed = 0.4*speed + 0.6*last_speed
        x = prev_x + speed*math.sin(phi)
        y = prev_y + speed*math.cos(phi)
        last_speed = speed
        last_phi = phi
        plot(prev_x, prev_y, x, y, 0.8)
        
        if (x <= 0):
            last_phi = random.uniform(0, math.pi)
        if (x >= 600):
            last_phi = random.uniform(math.pi, 2*math.pi) 
        if (y <= 0):
            last_phi = random.uniform(-0.5*math.pi, 0.5*math.pi)
        if (y >= 600):
            last_phi = random.uniform(0.5*math.pi, 1.5*math.pi)


        trainx_row = numpy.append(trainx_row, x)
        trainy_row = numpy.append(trainy_row, y)
        df_x.loc[i] = trainx_row
        df_y.loc[i] = trainy_row
        trainx_row = numpy.array([])
        trainy_row = numpy.array([])
        prev_x = x
        prev_y = y
        trainx_row = numpy.append(trainx_row, prev_x)
        trainx_row = numpy.append(trainx_row, prev_y)
        trainx_row = numpy.append(trainx_row, last_phi)
        trainx_row = numpy.append(trainx_row, last_speed)
        trainy_row = numpy.append(trainy_row, prev_x)
        trainy_row = numpy.append(trainy_row, prev_y)
        trainy_row = numpy.append(trainy_row, last_phi)
        trainy_row = numpy.append(trainy_row, last_speed)


df_x.to_csv('train_x_150.csv', index = False)
df_y.to_csv('train_y_150.csv', index = False)

plt.show()