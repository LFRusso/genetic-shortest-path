import numpy as np
import copy

def main():
    pass

if __name__ == '__main__':
    main()

# Partícula/Elemento representada pelo conjunto de pontos que deverão ficar no melhor caminho 
class Particle:
    def __init__(self, size, f, A, B):
        self.size = size    
        self.f = f
        self.fitness = 0
        
        # Iniciando partículas aleatóriamente
        p = np.zeros((size+2, 3))
        p[:, 0] = np.linspace(A[0], B[0], size+2)
        p[:, 1] = np.linspace(A[1], B[1], size+2)
        p[:, 2] = np.linspace(A[2], B[2], size+2)        
        rand = np.random.normal(scale=.1  ,size=(size, 3))

        p[1:-1] += rand
        self.state = np.array(p)
        
        self.pbest = self.state
        self.velocity = np.zeros(shape=(size, 3))
        return

    # Compute distances between points
    def distances(self):
        l_vec = np.zeros((self.state.shape[0]-1, 1))
        for i in range(1, self.state.shape[0]):
            l_vec[i-1] = np.linalg.norm(self.state[i-1] - self.state[i])

        return l_vec
    
    # Fitness da partícula
    def get_fitness(self, weights=[.8, 2.2, 1.5]):
        l_vec = self.distances()
        L = np.sum(l_vec)
        L_med = L / (self.state.shape[0] - 1) 

        # Total length
        loss_1 = L

        # Equal lengths
        loss_2 = np.sum(np.square(l_vec - L_med))

        # Constraint
        loss_3 = np.sum(np.square(self.f(self.state[:, 0], self.state[:, 1], self.state[:, 2])))

        loss = weights[0]*loss_1 + weights[1]*loss_2 + weights[2]*loss_3
        fit = 1/loss
        
        self.fitness = fit
        return fit
      
    # Dada uma taxa de mutação, realiza a mutação da partícula
    def mutate(self, rate=0.01):
        mutated_state = np.copy(self.state)
        if (np.random.random() < rate):
            #mutate
            rand_scale = np.absolute(np.random.normal(loc=0.01, scale=0.025))
            mutation = np.random.normal(scale=rand_scale, size=self.state.shape)
            mutated_state[1:-1] = self.state[1:-1] + mutation[1:-1]
        self.state = mutated_state
        return
        
    def show(self):
        for i in range(self.size):
            print(f"\t{self.state[i]}")
        return


# ================= Ḿodelo para obtenção do melhor caminho ===============
class Path:
    def __init__(self, points=5, n_particles=5, max_generations=1000, mutation_rate=0.1, crossover_p=0.6, n_best=2):
        self.n_particles = n_particles
        self.mutation_rate = mutation_rate
        self.n_best = n_best
        self.crossover_p = crossover_p
        self.max_generations = max_generations
        self.fitness = []
        self.dimension = points
        self.f = None
        return
    
    def crossover(self, P):
        size = self.n_particles
        crossed_P = np.zeros((size, P.shape[1], P.shape[2]))
        no_genes = P.shape[1]

        for i in range(size):
            if (np.random.random() < self.crossover_p):
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

    
    def fit(self, f, A, B):
        self.f = f
        self.A = A
        self.B = B
        
        # Inicializando partículas aleatóriamente
        particles = []
        for i in range(self.n_particles):
            new_particle = Particle(self.dimension, f, A, B)
            particles.append(new_particle)
        self.gbest = None
        self.particles = np.array(particles)
        
        # Passando função para partículas
        for i in range(self.n_particles):
            self.particles[i].f = self.f
        
        # Valor inicial de melhor global (não é importante, será atualizado em seguida)
        self.gbest = copy.deepcopy(self.particles[0])
        
        for t in range(self.max_generations):
            # obtendo novo melhor global
            for index,particle in enumerate(self.particles):
                if (self.gbest.get_fitness() < particle.get_fitness()):
                    self.gbest = copy.deepcopy(particle)
            self.fitness.append(self.gbest.get_fitness())
            
            # Atualizando estado das partículas
            
            # Obtendo lista de fitness das partítulas para serem ordenadas 
            fitness_list = np.zeros(self.n_particles)
            for index in range(self.n_particles):
                fitness_list[index] = self.particles[index].get_fitness()
            # Ordenando partículas pelo fitness em ordem decrescente
            sorted_idxs = fitness_list.argsort()
            self.particles = self.particles[sorted_idxs[::-1]]
            
            # Realizando crossover entre as n melhores partículas
            new_particles = np.array([p.state for p in copy.deepcopy(self.particles[:self.n_best])])
            new_particles = self.crossover(new_particles)

            # Atualizando estados das partículas
            for idx in range(self.n_particles):
                self.particles[idx].state = new_particles[idx]
                # Realizando mutação das partículas
                self.particles[idx].mutate(self.mutation_rate)

        return self.fitness[-1]
    
    def show(self):
        for i in range(self.n_particles):
            print(i)
            self.particles[i].show()
        return
