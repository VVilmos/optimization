from docplex.mp.model import Model
import random
import math
import pandas
import matplotlib
import matplotlib.pyplot as plt
import numpy 
import time


model = Model(name = "connectingUEs")
c = 200
w = 500



#Location and energy cost of gNBs
gNodes_x = [100, 300, 500, 100, 300, 500, 100, 300, 500]
gNodes_y = [100, 100, 100, 300, 300, 300, 500, 500, 500]
cost = [6,6,6,6,6,6,6,13,6]

#Location of users is random
b = []
users_x = numpy.array([])
users_y = numpy.array([])
for u in range(90):
    x = random.randint(1, 200 )
    y = random.randint(1, 200)
    bw = random.randint(1, 10)
    users_x = numpy.append(users_x, x)
    users_y = numpy.append(users_y, y)
    b.append(bw)



distance = []
#calculating the distance of every UE to every gNB
def calculate_distance():
    for g in range(9):
        line = []
        for u in range(90):
            d = math.sqrt(pow(gNodes_x[g] - users_x[u], 2) + pow(gNodes_y[g] - users_y[u],2))
            line.append(d)
        distance.append(line)

last_speed = []
last_phi = []
#initializing the users' speed with random values
#their direction is random but with a bias towards the right upper quadrant
def init_user():
    for u in range(90):
        phi = random.uniform(0, 0.5*math.pi)
        speed = random.randint(1, 40)
        last_speed.append(speed)
        last_phi.append(phi)


#random walk mobility model
#the user moves in the direction of the last movement with 60% probability
def randomwalk(list_x, list_y):
    for u in range(90):
        phi = random.uniform(last_phi[u] - 0.25*math.pi, last_phi[u] + 0.25*math.pi)
        speed = random.randint(1, 40)
        phi = 0.4*phi + 0.6*last_phi[u]
        speed = 0.4*speed + 0.6*last_speed[u]
        list_x[u] += speed*math.cos(phi)
        list_y[u] += speed*math.sin(phi)
        last_speed[u] = speed
        last_phi[u] = phi
        numpy.clip(list_x, 10, 590, out=list_x)
        numpy.clip(list_y,10, 590, out=list_y)
      


fig, ax = plt.subplots()   
#plotting the location of users, gNBs and their association
def plot(i):
    plt.ion()
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 600)
    for i in range(i):
       for g in range(9):
            for u in range(90):
                if (solution[x[g,u]]):
                    xp = [gNodes_x[g], users_x[u]]
                    yp = [gNodes_y[g], users_y[u]]
                    ax.plot(xp, yp, c='grey', linewidth=0.5, zorder=1)
 
       ax.scatter(users_x, users_y, s=25)

       for g in range(9):
            if (solution[y[g]]):
                ax.scatter(gNodes_x[g], gNodes_y[g], c='red', marker='^', s=150, zorder=2)     
            else:
                ax.scatter(gNodes_x[g], gNodes_y[g], c='green',marker='^',  s=150, zorder=2)

       plt.draw()
       plt.pause(0.5)
       ax.clear()
       ax.set_xlim(0, 600)
       ax.set_ylim(0, 600)
       randomwalk(users_x,users_y)

    


#decision variables
x = model.binary_var_matrix(range(9), range(90), name = "x")
y = model.binary_var_list(range(9), name = "y")



#constraints
for u in range(90):
    model.add_constraint(model.sum(x[g, u] for g in range(9)) == 1)

for g in range(9):
    for u in range(90):
        model.add_constraint(x[g, u] <= y[g])

for g in range(9):
    model.add_constraint(model.sum(x[g, u]*b[u] for u in range(90)) <= c, ctname = f'gNB capacity_{g}')

#initializing the users' speed and direction
init_user()
#run optimization #1
calculate_distance()
model.minimize(model.sum(x[g, u]*distance[g][u] for g in range(9) for u in range(90)) + model.sum(w*y[g]*cost[g] for g in range(9)))
solution = model.solve()
#plotting for 5 seconds
plot(10)

#run optimization #2
distance.clear()
calculate_distance()
model.minimize(model.sum(x[g, u]*distance[g][u] for g in range(9) for u in range(90)) + model.sum(w*y[g]*cost[g] for g in range(9)))
solution = model.solve()
plot(10)

#run optimization #3
distance.clear()
calculate_distance()
model.minimize(model.sum(x[g, u]*distance[g][u] for g in range(9) for u in range(90)) + model.sum(w*y[g]*cost[g] for g in range(9)))
solution = model.solve()
plot(10)


plt.ioff()
plt.show()