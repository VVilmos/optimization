import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import numpy as np
users_x = np.array([])
users_y = np.array([])

last_speed = []
last_phi = []


df = pd.DataFrame()
num_rows = 90

#initializing the users' location, speed and direction with random values
for u in range(90):
    phi = random.uniform(0, 2*math.pi)
    speed = random.randint(1, 40)
    last_speed.append(speed)
    last_phi.append(phi)
    x = random.uniform(1, 600)
    y = random.uniform(1, 600)
    users_x = np.append(users_x, x)
    users_y = np.append(users_y, y)
    colname_x = 'user' + str(u) + '_x'
    colname_y = 'user' + str(u) + '_y'   
    df.loc[0, colname_x] = x
    df.loc[0, colname_y] = y
    #the previous four lines do not work as expected beacause:
    #1. the column names are the same
    #2. the values are overwritten



#random walk mobility model
#the user moves in the direction of the last movement with 60% probability
def randomwalk(list_x, list_y, df):
    newlocation = []
    for i in range(1000):
        for u in range(90):
            
            phi = random.uniform(last_phi[u] - 0.25*math.pi, last_phi[u] + 0.25*math.pi)
            speed = random.randint(1, 20)
            phi = 0.4*phi + 0.6*last_phi[u]
            speed = 0.4*speed + 0.6*last_speed[u]
            list_x[u] += speed*math.cos(phi)
            list_y[u] += speed*math.sin(phi)
            last_speed[u] = speed
            last_phi[u] = phi
            np.clip(list_x, 0, 600, out=list_x)
            np.clip(list_y,0, 600, out=list_y)
            if (list_x[u] == 0 or list_x[u] == 600 or list_y[u] == 0 or list_y[u] == 590):
                last_phi[u] = random.uniform(0, 2*math.pi)
            df.loc[i, 'user' + str(u) + '_x'] = list_x[u]
            df.loc[i, 'user' + str(u) + '_y'] = list_y[u]
        newlocation.clear()



randomwalk(users_x, users_y, df)

print(df)
df.to_csv('predefinedwalk.csv', index=False)



