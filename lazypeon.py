from lazypeon import Path
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Preparação da figura
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

# Função de uma esfera
def f(x, y, z):
    # Sphere
    f = x**2 + y**2 + z**2 - 1
    return f

# Construção do plot da isosuperfície (meshgrid + marching cubes) e seleção dos dois pontos 
x, y, z = np.mgrid[-1:1:100j, -1:1:100j, -1:1.:100j]
fig = buildFig(x, y, z)
(A, B) = np.array([[0, 0, 1], [np.sqrt(1/2), np.sqrt(1/2), 0]])

path = Path(max_generations=1000, mutation_rate=0.2, crossover_p=0.7, points=10, n_particles=20, n_best=2)
fitness = path.fit(f, A, B)
print(fitness)

# Plot da evolução do fitness
#plt.plot(np.arange(0, len(path.fitness)), path.fitness)
#plt.show()

# Plot de todas as partículas
for p in [particle.state for particle in path.particles]:
   fig.add_trace(go.Scatter3d(
       x = p[:,0],
       y = p[:,1],
       z = p[:,2],
       line=dict(
           color='red',
           width=1
           )
       ))

# Plot da partícula com melhor fitness
p = path.gbest.state
fig.add_trace(go.Scatter3d(
   x = p[:,0],
   y = p[:,1],
   z = p[:,2],
   line=dict(
       color='green',
       width=2
       )
   ))
fig.show()