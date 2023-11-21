from docplex.mp.model import Model
import random
import math
import pandas
import matplotlib
import matplotlib.pyplot as plt
import numpy 


df = pandas.DataFrame()

#reading the data from the csv file
df = pandas.read_csv('predefinedwalk.csv')
#Location and energy cost of gNBs
gNodes_x = [100, 300, 500, 100, 300, 500, 100, 300, 500]
gNodes_y = [100, 100, 100, 300, 300, 300, 500, 500, 500]
cost = [6,6,6,6,6,6,6,6,6]

b = []
users_x = numpy.array([])
users_y = numpy.array([])
#Location of users is read from the csv file
#bandwidth of users is randomly generated
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


#simulating the movement of users for 10 steps
def plot(i):
    for i in range((i-1)*10, 10*i):
       #changing the location of users from the csv file:
       for u in range(90):
            users_x[u] = df.loc[i, 'user' + str(u) + '_x']
            users_y[u] = df.loc[i, 'user' + str(u) + '_y']
       
    
#modelling the problem
model = Model(name = "connectingUEs")
c = 150
power_cost_weight = 500
switching_cost_weight = 0
distance_weight = 1
handover_cost_weight = 0
previous_state_y = []
previous_state_x = [[]]
fig, ax = plt.subplots()
n = []
db = 0
#all gNBs are turned on at the beginning
for g in range(9):
    previous_state_y.append(0)

#all users are connected to the first gNB at the beginning
for g in range(9):
    previous_state_x.append([])
    for u in range(90):
        previous_state_x[g].append(0)


#creating a model with the previous state of gNBs
#every optimization is done on a new model with the updated previous state
def model_update(previous_state_y):
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


#calculating the number of user handovers after the last optimization
def calculate_handovers(solution):
    handovers = 0
    for g in range(9):
        for u in range(90):
            if (solution[dx[g, u]] != 0):
                handovers += 1
                continue
    return handovers
    


# running 3 sets of 100 optimizations with different switching cost weights
for k in range(3):
    for i in range(1, 101):
        distance.clear()
        calculate_distance()
        model, x, y, dy, dx = model_update(previous_state_y=previous_state_y)
        #objective function
        model.minimize(model.sum(distance_weight*x[g, u]*distance[g][u] for g in range(9) for u in range(90)) + model.sum(power_cost_weight*y[g]*cost[g] for g in range(9)) + model.sum(switching_cost_weight*dy[g] for g in range(9)) + handover_cost_weight*model.sum(dx[g, u] for g in range(9) for u in range(90))/2)
        solution = model.solve()
        db += calculate_handovers(solution)
        n.append(db)
        #updating the previous state of gNBs
        for g in range(9):
            previous_state_y[g] = solution[y[g]]

        #updating the previous state of users
        for g in range(9):
            for u in range(90):
                previous_state_x[g][u] = solution[x[g, u]]
        plot(i)


    ax.plot(n, label = 'Handover cost =' + str(handover_cost_weight))

    handover_cost_weight += 12*(k+1)
    users_x = numpy.array([])
    users_y = numpy.array([])
    for u in range(90):
        x = df.loc[0, 'user' + str(u) + '_x']
        y = df.loc[0, 'user' + str(u) + '_y']
        users_x = numpy.append(users_x, x)
        users_y = numpy.append(users_y, y)


    db = 0
    n.clear()

plt.legend()
plt.xlabel('Number of optimizations')
plt.ylabel('Number of handovers')
plt.show()

