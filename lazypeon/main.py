import numpy as np
import plotly.graph_objects as go

# Function to plot
def f(x, y, z):
    # Sphere
    #f = x**2 + y**2 + z**2 - 4
    
    # <3
    f = (x**2 + (9*y**2)/4 + z**2 - 1)**3 - (x**2)*z**3 - (9*(y**2)*(z**3))/80
    
    # Surface
    #f = x**2 * y**2 + z
    #f = x**2 - y**2 - z
    
    # Cone
    #f = (x**2 + y**2) - (z - 5)**2

    return f

def genPoints(n, A, B, gen_size=5):
    p = np.zeros((n+2, 3))
    p[:, 0] = np.linspace(A[0], B[0], n+2)
    p[:, 1] = np.linspace(A[1], B[1], n+2)
    p[:, 2] = np.linspace(A[2], B[2], n+2)
    
    rand = np.random.normal(scale=.1  ,size=(gen_size, n, 3))
    
    P = np.array(gen_size*[p])
    P[:, 1:-1] += rand
    
    return P

def crossover(P, size, probability=0.6):
    crossed_P = np.zeros((size, P.shape[1], P.shape[2]))
    no_genes = P.shape[1]

    for i in range(size):
        if (probability < np.random.random()):
            # crossover
            (idx1, idx2) = (np.random.choice(P.shape[0]), np.random.choice(P.shape[0]))
            
            
            (p1, p2) = (P[idx1], P[idx2])
            cross_idx = np.random.randint(2, no_genes)
            crossed_P[i] = np.vstack((p1[:cross_idx], p2[cross_idx:]))
        else:
            # copy a random parent
            idx = np.random.choice(P.shape[0])
            crossed_P[i] = P[idx]        
    
    return crossed_P

def mutate(P, p=0.02):
    mutated_P = np.copy(P)
    for i in range(P.shape[0]):
        if (np.random.random() < p):
            #mutate
            rand_scale = np.absolute(np.random.normal(loc=0.01, scale=0.025))
            mutation = np.random.normal(scale=rand_scale, size=P[i].shape)
            mutated_P[i, 1:-1] = P[i, 1:-1] + mutation[1:-1]
    return mutated_P

def evolve(P):
    size = P.shape[0]
    losses = np.zeros(size)
    for i in range(size):
        losses[i] = loss(P[i], [.8, 2.2, 1.5])
    #print("Loss = ", np.average(losses))
    
    sorted_idxs = losses.argsort()
    P = P[sorted_idxs]
    print(np.average(losses), np.sort(losses)[0])
    
    new_P = P[:2]
    new_P = crossover(new_P, size, 0.3)
    mutated_P = mutate(new_P, .2)
    ##
    mutated_P[0] = P[0]

    return mutated_P

# Compute distances between points
def distances(p):
    l_vec = np.zeros((p.shape[0]-1, 1))
    for i in range(1, p.shape[0]):
        l_vec[i-1] = np.linalg.norm(p[i-1] - p[i])

    return l_vec

# Compute loss for generation
def loss(p, weights=[1., .9, 1.1]):
    l_vec = distances(p)
    L = np.sum(l_vec)
    L_med = L / (p.shape[0] - 1) 
   
    # Total length
    loss_1 = L
    
    # Equal lengths
    loss_2 = np.sum(np.square(l_vec - L_med))
    
    # Constraint
    loss_3 = np.sum(np.square(f(p[:, 0], p[:, 1], p[:, 2])))
    
    loss = weights[0]*loss_1 + weights[1]*loss_2 + weights[2]*loss_3
    return loss

def buildFig(x, y, z):
    fig = go.Figure(data=go.Isosurface(
        x = x.flatten(),
        y = y.flatten(),
        z = z.flatten(),
        value = f(x, y, z).flatten(),
        isomin = 0,
        isomax = 0,
        opacity = 0.25,
        colorscale ='BlueRed',
        showscale = False
        ))
    
    fig.update_layout(
        margin=dict(t=0, l=0, b=0), #tight layout
        )
    
    return fig

x, y, z = np.mgrid[-2:2:100j, -2:2:100j, -2:2.:100j]
fig = buildFig(x, y, z)
#(A, B) = np.array([[-1.9, 1.3, 1.92], [2.1, 1.2, 2.97]])
#(A, B) = np.array([[-1, -1, 3], [0, 4, 1]])
(A, B) = np.array([[0, 0, -1], [0.5, 0.05, 1.2]])
P = genPoints(20, A, B)

for i in range(50000):
    P = evolve(P)
    #print("Generation", i)

#for p in P:
#   fig.add_trace(go.Scatter3d(
#       x = p[:,0],
#       y = p[:,1],
#       z = p[:,2],
#       line=dict(
#           color='red',
#           width=2
#           )
#       ))


p = P[0]
fig.add_trace(go.Scatter3d(
   x = p[:,0],
   y = p[:,1],
   z = p[:,2],
   line=dict(
       color='red',
       width=2
       )
   ))

fig.show()
