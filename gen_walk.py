import pandas
import random
import math
import numpy

filename = "walk_50.csv"
df = pandas.DataFrame(numpy.empty((9000,2), dtype = float)) 
df.columns = ['x', 'y']
start_x = random.uniform(0 , 600)
start_y = random.uniform(0 , 600)

last_speed = random.randint(1, 50)
last_phi = random.uniform(0, 2*math.pi)

def randomwalk(start_x, start_y, last_phi, last_speed):
    prev_x = start_x
    prev_y = start_y
    for i in range(9000):
        phi = random.uniform(last_phi - 0.25*math.pi, last_phi + 0.25*math.pi)
        speed = random.randint(1, 50)
        phi = 0.4*phi + 0.6*last_phi
        speed = 0.4*speed + 0.6*last_speed
        x = prev_x + speed*math.sin(phi)
        y = prev_y + speed*math.cos(phi)

        df.loc[i] = [x, y]
        if (x <= 0):
            phi = random.uniform(0, math.pi)
        if (x > 600):
            phi = random.uniform(math.pi, 2*math.pi)
        if (y <= 0):
            phi = random.uniform(-0.5*math.pi, 0.5*math.pi)
        if (y >= 600):
            phi = random.uniform(0.5*math.pi, 1.5*math.pi)


        last_speed = speed
        last_phi = phi
        prev_x = x
        prev_y =y

randomwalk(start_x, start_y, last_phi, last_speed)
df.to_csv(filename, index=False)