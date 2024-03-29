from docplex.mp.model import Model
import random
import math
import pandas
import matplotlib
import matplotlib.pyplot as plt
import numpy 

list_x = numpy.array([])
list_y = numpy.array([])

df = pandas.DataFrame()

#reading the data from the csv file
df = pandas.read_csv('predefinedwalk.csv')
#Location and energy cost of gNBs
gNodes_x = [100, 300, 500, 100, 300, 500, 100, 300, 500]
gNodes_y = [100, 100, 100, 300, 300, 300, 500, 500, 500]
cost = [6,6,6,6,6,6,6,6,6]

#Location of users is read from the csv file
# bandwidth of users is randomly generated
b = []
users_x = numpy.array([])
users_y = numpy.array([])
for u in range(90):
    x = df.loc[0, 'user' + str(u) + '_x']
    y = df.loc[0, 'user' + str(u) + '_y']
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


fig, ax = plt.subplots()   
#plotting the location of users, gNBs and their association
def plot(i):
    plt.ion()
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 600)
    for i in range((i-1)*10, 10*i):
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
            if (solution[dy[g]] is not 0):
                if (solution[y[g]] is not 0):
                    circle= plt.Circle((gNodes_x[g], gNodes_y[g]-4), 20, color='green', fill=False, linestyle = 'dashed')
                    ax.add_artist(circle)
                else:
                    circle = plt.Circle((gNodes_x[g], gNodes_y[g]-4), 20, color='red', fill=False, linestyle = 'dashed')
                    ax.add_artist(circle)
       plt.draw()
       plt.pause(0.5)
       ax.clear()
       ax.set_xlim(0, 600)
       ax.set_ylim(0, 600)
       #changing the location of users from the csv file:
       for u in range(90):
            users_x[u] = df.loc[i, 'user' + str(u) + '_x']
            users_y[u] = df.loc[i, 'user' + str(u) + '_y']
       
    
#modelling the problem
model = Model(name = "connectingUEs")
c = 150  #capacity of gNBs
power_cost_weight = 500
switching_cost_weight = 1000
handover_cost_weight = 10
distance_weight = 10


previous_state_y = []
previous_state_x = [[]]

#all gNBs are turned off at the beginning
for g in range(9):
    previous_state_y.append(0)

#all users are associated with the first gNB at the beginning
#fill the matrix with 0s
for g in range(9):
    previous_state_x.append([])
    for u in range(90):
        previous_state_x[g].append(0)



#number of gNBs that are switched on or off
n = 0

def model_update(previous_state_y, previous_state_x):
    model = Model(name = "connectingUEs")

    #decision variables
    x = model.binary_var_matrix(range(9), range(90), name = "x")
    y = model.binary_var_list(range(9), name = "y")
    #dy is representing the number of gNBs that are turned on or off after the optimization
    dy = model.binary_var_list(range(9), name = "dy")
    #dx is representing the number of user handovers after the optimization
    dx = model.binary_var_matrix(range(9), range(90), name = "dx")

    #constraints
    for g in range(9):
        model.add_constraint(y[g]-previous_state_y[g] <= dy[g])
        model.add_constraint(previous_state_y[g]-y[g] <= dy[g])
        model.add_constraint(y[g] + previous_state_y[g] >=  dy[g])
        model.add_constraint(2-y[g]-previous_state_y[g] >= dy[g])

    for g in range(9):
        for u in range(90):
            model.add_constraint(x[g, u] - previous_state_x[g][u] <= dx[g, u])
            model.add_constraint(previous_state_x[g][u] - x[g, u] <= dx[g, u])
            model.add_constraint(x[g, u] + previous_state_x[g][u] >= dx[g, u])
            model.add_constraint(2-x[g, u] - previous_state_x[g][u] >= dx[g, u])

    for u in range(90):
        model.add_constraint(model.sum(x[g, u] for g in range(9)) == 1)

    for g in range(9):
        for u in range(90):
            model.add_constraint(x[g, u] <= y[g])

    for g in range(9):
        model.add_constraint(model.sum(x[g, u]*b[u] for u in range(90)) <= c, ctname = f'gNB capacity_{g}')

    return model, x, y, dy, dx
    
#run 100 optimization
for i in range(1, 101):
    distance.clear()
    calculate_distance()
    model, x, y, dy, dx = model_update(previous_state_y=previous_state_y, previous_state_x=previous_state_x)
    model.minimize(model.sum(distance_weight*x[g, u]*distance[g][u] for g in range(9) for u in range(90)) + model.sum(power_cost_weight*y[g]*cost[g] for g in range(9)) + model.sum(switching_cost_weight*dy[g] for g in range(9)) + model.sum(handover_cost_weight*dx[g, u] for g in range(9) for u in range(90))/2)
    solution = model.solve()

    for g in range (9):
        n += solution[dy[g]]
        print(solution[dy[g]], end = ", ")
    print()

    #updating the previous state of gNBs
    for g in range(9):
        previous_state_y[g] = solution[y[g]]

    #upfadting the previous state of users
    for g in range(9):
        for u in range(90):
            previous_state_x[g][u] = solution[x[g, u]]
    plot(i)

print("number of gNB switches: ", n)
plt.ioff()
plt.show()
