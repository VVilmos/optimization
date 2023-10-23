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
switching_cost_weight = 10
distance_weight = 1
previous_state = []

fig, ax = plt.subplots()
n = []
db = 0
#all gNBs are turned on at the beginning
for g in range(9):
    previous_state.append(0)

#creating a model with the previous state of gNBs
#every optimization is done on a new model with the updated previous state
def model_update(previous_state):
    model = Model(name = "connectingUEs")

    #decision variables
    x = model.binary_var_matrix(range(9), range(90), name = "x")
    y = model.binary_var_list(range(9), name = "y")
    #dy is representing the number of gNBs that are turned on or off after the optimization
    dy = model.binary_var_list(range(9), name = "dy")

    #constraints
    for g in range(9):
        model.add_constraint(y[g]-previous_state[g] <= dy[g])
        model.add_constraint(previous_state[g]-y[g] <= dy[g])
        model.add_constraint(y[g] + previous_state[g] >=  dy[g])
        model.add_constraint(2-y[g]-previous_state[g] >= dy[g])
    for u in range(90):
        model.add_constraint(model.sum(x[g, u] for g in range(9)) == 1)

    for g in range(9):
        for u in range(90):
            model.add_constraint(x[g, u] <= y[g])

    for g in range(9):
        model.add_constraint(model.sum(x[g, u]*b[u] for u in range(90)) <= c, ctname = f'gNB capacity_{g}')

    return model, x, y, dy

#calculating the number of gNBs that are turned on or off after the last optimization
def numberofswitching(solution, previous_state):
    n = 0
    for g in range(9):
        if (solution[y[g]] != previous_state[g]):
            n += 1
    return n
    


# running 3 sets of 100 optimizations with different switching cost weights
for k in range(3):
    for i in range(1, 101):
        distance.clear()
        calculate_distance()
        model, x, y, dy = model_update(previous_state=previous_state)
        model.minimize(model.sum(distance_weight*x[g, u]*distance[g][u] for g in range(9) for u in range(90)) + model.sum(power_cost_weight*y[g]*cost[g] for g in range(9)) + model.sum(switching_cost_weight*dy[g] for g in range(9)))
        solution = model.solve()
        db += numberofswitching(solution, previous_state)
        n.append(db)
        #updating the previous state of gNBs
        for g in range(9):
            previous_state[g] = solution[y[g]]
        plot(i)

    ax.plot(n, label = 'Switching cost =' + str(switching_cost_weight))

    switching_cost_weight *= 7
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
plt.ylabel('Number of gNB switching')
plt.show()

