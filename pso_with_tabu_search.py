import random
import math
import numpy as np
import pandas as pd
from queue import deque



SWARM_SIZE = 20
DIMENTIONS = 20
ITERATIONS = 1000
PSO_RUNS = 30
ENABLE_EXEC_LOG = True
TS_MEMORY = 30
ENABLE_TS_CACHE = True



class PSO:
    """
    Particle Swarm Optimization implementation with star topology
    """
    def __init__(self, w, c1, c2, bounds, obj_func):
        """
        POS random initialization of particle positions and velocities
        """
        self.inertia = w
        self.cognitive = c1
        self.social = c2
        self.obj_func = obj_func
        # establish the swarm
        swarm=[]
        for i in range(SWARM_SIZE):
            start_position = [random.uniform(bounds[0], bounds[1]) for i in range(DIMENTIONS)]
            swarm.append(Particle(start_position))
        self.swarm = swarm
        self.bounds = bounds

    def run(self):
        num_dimensions = DIMENTIONS
        err_best_g = -1                   # best error for group
        pos_best_g = []                   # best position for group
        swarm = self.swarm

        # begin optimization loop
        for i in range(ITERATIONS):
            # cycle through particles in swarm and evaluate fitness
            for j in range(SWARM_SIZE):
                swarm[j].evaluate(self.obj_func)

                # determine if current particle is the best (globally)
                if swarm[j].particle_err < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].particle_pos)
                    err_best_g=float(swarm[j].particle_err)

            # cycle through swarm and update velocities and position
            for j in range(SWARM_SIZE):
                swarm[j].update_velocity(pos_best_g, self.inertia, self.cognitive, self.social)
                swarm[j].update_position(self.bounds)

        return err_best_g



class Particle:
    """
    Represents a particle of the swarm
    """
    def __init__(self, start):
        self.particle_pos = []          # particle position
        self.particle_vel = []          # particle velocity
        self.best_pos = []              # best position individual
        self.best_err = -1              # best result individual
        self.particle_err = -1          # result individual
        
        for i in range(DIMENTIONS):
            self.particle_vel.append(random.uniform(-1,1))
            self.particle_pos.append(start[i])

    def evaluate(self, costFunc):
        """
        Evaluate current fitness
        """
        self.particle_err = costFunc(self.particle_pos)

        # check to see if the current position is an individual best
        if self.particle_err < self.best_err or self.best_err == -1:
            self.best_pos = self.particle_pos
            self.best_err = self.particle_err

    def update_velocity(self, pos_best_g, w, c1, c2):
        """
        Updates new particle velocity
        """
        for i in range(DIMENTIONS):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.best_pos[i] - self.particle_pos[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.particle_pos[i])
            self.particle_vel[i] = w * self.particle_vel[i] + vel_cognitive + vel_social

    def update_position(self, bounds):
        """
        Updates the particle position based off new velocity updates
        """
        for i in range(DIMENTIONS):
            self.particle_pos[i] = self.particle_pos[i] + self.particle_vel[i]

            # adjust maximum position if necessary
            if self.particle_pos[i] > bounds[1]:
                self.particle_pos[i] = bounds[1]

            # adjust minimum position if neseccary
            if self.particle_pos[i] < bounds[0]:
                self.particle_pos[i] = bounds[0]



class ObjFn:
    """
    Objective functions for optimization with PSO
    """
    spherical_bounds = (-5.12, 5.12)
    ackley_bounds = (-32.768, 32.768)
    michalewicz_bounds = (np.finfo(np.float64).tiny, math.pi)
    katsuura_bounds = (-100, 100)

    def spherical(x):
        result = 0
        for i in range(len(x)):
            result = result + x[i]**2
        return result

    def ackley(x):
        n = len(x)
        res_cos = 0
        for i in range(n):
            res_cos = res_cos + math.cos(2*(math.pi)*x[i])
        #returns the point value of the given coordinate
        result = -20 * math.exp(-0.2*math.sqrt((1/n)*ObjFn.spherical(x)))
        result = result - math.exp((1/n)*res_cos) + 20 + math.exp(1)
        return result

    def michalewicz(x):
        m=10
        result = 0
        for i in range(len(x)):
            result += math.sin(x[i]) * (math.sin((i*x[i]**2)/(math.pi)))**(2*m)
        return -result

    def katsuura(x):
        prod = 1
        d = len(x)
        for i in range(0, d):
            sum = 0
            two_k = 1
            for k in range(1, 33):
                two_k = two_k * 2
                sum += abs(two_k * x[i] - round(two_k * x[i])) / two_k
            prod *= (1 + (i+1) * sum)**(10/(d**1.2))
        return (10/(d**2)) * (prod - 1)



class BenchmarkPSO:
    """
    Implements the logical steps of the PSO Intelligent Parameter Tuning
    """
    def __init__(self):
        next

    def run_benchmark():
        if ENABLE_EXEC_LOG:
            print("******** RUNNING PSO BENCHMARKS ********")
            print("\nBelow are logged sample optimization results:")
            print("\tW\tC1\tC2\tSpherical\t\tAckley\t\tMichalewicz\t\tKatsuura")
        data = []
        for w in np.arange(-1.1, 1.15, 0.1):
            for c in np.arange(0.05, 2.525, 0.05):
                s_res = BenchmarkPSO.run_pso_batch(w, c, c, ObjFn.spherical, ObjFn.spherical_bounds)
                a_res = BenchmarkPSO.run_pso_batch(w, c, c, ObjFn.ackley, ObjFn.ackley_bounds)
                m_res = BenchmarkPSO.run_pso_batch(w, c, c, ObjFn.michalewicz, ObjFn.michalewicz_bounds)
                k_res = BenchmarkPSO.run_pso_batch(w, c, c, ObjFn.katsuura, ObjFn.katsuura_bounds)
                data.append([w, c, c, s_res, a_res, m_res, k_res])
                if ENABLE_EXEC_LOG and math.floor(c * 100) % 25 == 0:
                    print("\t{:.1f}\t{:.2f}\t{:.2f}\t{:.10f}\t{:.10f}\t\t{:.10f}\t{:.10f}".format(w, c, c, s_res, a_res, m_res, k_res))
        df = pd.DataFrame(data, columns=['inertia', 'cognitive', 'social', 'spherical', 'ackley', 'michalewicz', 'katsuura'])
        if ENABLE_EXEC_LOG:
            print("PSO Benchmarking completed")
        return df
        
    def run_pso_batch(w, c1, c2, func, bounds):
        total = 0
        for i in range(PSO_RUNS):
            pso = PSO(w = w, c1 = c1, c2 = c2, bounds = bounds, obj_func = ObjFn.spherical)
            total += pso.run()
        return total / PSO_RUNS



class TabuSearch:
    """
    Tabu Search algorithm implementation to find the optimal w, c1, c2 coefficients
    for a PSO run
    """
    def __init__(self, objective_function, boundaries, TS_MOVES):
        """
        Initialization method
        """
        self.tabuMemory = deque(maxlen = TS_MEMORY)
        self.func = objective_function
        self.bounds = boundaries
        # if ENABLE_TS_CACHE:
        self.cache = dict()
        self.moves = TS_MOVES

    def solutionFitness(self, solution):
        """
        Solution fitness score is calculated as a total of all information gains for each selected feature
        """
        if ENABLE_TS_CACHE:
            if not solution in self.cache.keys():
                pso = PSO(w = solution[0], c1 = solution[1], c2 = solution[2], obj_func = self.func, bounds = self.bounds)
                self.cache[solution] = pso.run()
            return self.cache.get(solution)
        pso = PSO(w = solution[0], c1 = solution[1], c2 = solution[2], obj_func = self.func, bounds = self.bounds)           
        return pso.run()

    def memorize(self, solution):
        """
        Memorizes current solution for further verification of tabu and aspiration criterias
        """
        self.tabuMemory.append(";".join(format(f, '.1f') for f in solution))

    def tabuCriteria(self, solution):
        """
        Verifyes if a solution is not tabooed
        """
        str = ";".join(format(f, '.1f') for f in solution)
        if str in self.tabuMemory:
            # if ENABLE_EXEC_LOG:
            #     print("Tabooed coefficients: {}".format(str))
            return False
        return True

    def putativeNeighbors(self, solution):
        """
        Find coefficients values within 0.1 step on any coefficient that satisfy tabu criteria and order-1/-2 stability
        """
        neighbors = list()
        for a in [-0.1, 0, 0.1]:
            for b in [-0.1, 0, 0.1]:
                for c in [-0.1, 0, 0.1]:
                    n = list(solution).copy()
                    n[0] += a
                    n[1] += b
                    n[2] += c
                    if TabuSearch.stabilityCheck(n) and self.tabuCriteria(n):
                        neighbors.append(tuple(n))
        return neighbors

    
    
    def stabilityCheck(solution):
        """
        Verify if the solution satisfies the condition for order-1 and order-2 stability
        """
        w = solution[0]
        c1 = solution[1]
        c2 = solution[2]
        return abs(w) < 1 and c1 + c2 > 0 and c1 + c2 < 24 * (1 - w**2) / (7 - 5*w)

    def randomSolution():
        """
        Initialize first solution randomly, such that it satisfies stability condition
        """
        rand_velocity = random.randint(4, 9) / 10 #random.randint(-5, 5) / 10
        rand_cognitive = random.randint(11, 17)/10  #random.randint(-15, 15) / 10
        rand_social = random.randint(11, 17)/10  #random.randint(-15, 15) / 10
        return (rand_velocity, rand_cognitive, rand_social)

    def run(self, init_solution):
        """
        Performs a run of Tabu Search based on initialized objective function and finds the optimal
        set of coefficients for the Particle Swarm Optimization algorithm
        """
        self.best_solution = init_solution
        self.memorize(self.best_solution)
        self.curr_solution = self.best_solution

        for i in range(self.moves):
            if i in [300, 600, 900, 1200, 1500, 1800]:
                print("At step {}".format(i))
            neighbors = self.putativeNeighbors(self.curr_solution)
            bestFit = np.finfo(np.float64).max
            # Find the best solution from putative neighbors and makes it the current one
            for solution in neighbors:
                if self.solutionFitness(solution) < bestFit:
                    self.curr_solution = solution
                    bestFit = self.solutionFitness(solution)
            # Memorize the current solution
            self.memorize(self.curr_solution)
            # Verify if current solution is better than the best one, and saves the current as best, if true
            if self.solutionFitness(self.curr_solution) < self.solutionFitness(self.best_solution):
                self.best_solution = self.curr_solution
                if ENABLE_EXEC_LOG:
                    print("Next best solution ({}) on step {};\tResult: {:.10f}\t cache size: {}".format(" ".join(format(f, '.1f') for f in self.best_solution), i, bestFit, len(self.cache)))
        print(len(self.cache.keys()))
        # Return the best solution found
        res = self.solutionFitness(self.best_solution)
        return self.best_solution + (res,)






def main():
    # df = BenchmarkPSO.run_benchmark() # Thses commands to store benchmark function parameters
    # df.to_csv('pso_benchmark.csv')
    
    """
    TS_MOVES is our control parameter
    """
    print("Test on Spherical Function:\n")
    init_solution = TabuSearch.randomSolution()
    print("Initial coefficients: {}".format(init_solution))

    s_ts = TabuSearch(objective_function = ObjFn.spherical, boundaries = ObjFn.spherical_bounds, TS_MOVES = 20) 
    s_res = s_ts.run(init_solution)
    print("Spherical Best performing coefficients: w={:.1f} c1={:.1f} c2={:.1f}\nPSO result: {:.20f}".format(s_res[0], s_res[1], s_res[2], s_res[3]))
    
    print("Test on Ackley Function:\n")
    init_solution = TabuSearch.randomSolution()
    print("Initial coefficients: {}".format(init_solution))


    a_ts = TabuSearch(objective_function = ObjFn.ackley, boundaries = ObjFn.ackley_bounds, TS_MOVES = 20)
    a_res = a_ts.run(init_solution)
    print("Ackley Best performing coefficients: w={:.1f} c1={:.1f} c2={:.1f}\nPSO result: {:.20f}".format(a_res[0], a_res[1], a_res[2], a_res[3]))
    
    print("Test on Michalewicz Function:\n")
    init_solution = TabuSearch.randomSolution()
    print("Initial coefficients: {}".format(init_solution))


    m_ts = TabuSearch(objective_function = ObjFn.michalewicz, boundaries = ObjFn.michalewicz_bounds, TS_MOVES = 20)
    m_res = m_ts.run(init_solution)
    print("Michalewicz Best performing coefficients: w={:.1f} c1={:.1f} c2={:.1f}\nPSO result: {:.20f}".format(m_res[0], m_res[1], m_res[2], m_res[3]))
    
    print("Test on Katsuura Function:\n")
    init_solution = TabuSearch.randomSolution()
    print("Initial coefficients: {}".format(init_solution))


    k_ts = TabuSearch(objective_function = ObjFn.katsuura, boundaries = ObjFn.katsuura_bounds, TS_MOVES = 80)
    k_res = k_ts.run(init_solution)
    print("katsuura Best performing coefficients: w={:.1f} c1={:.1f} c2={:.1f}\nPSO result: {:.20f}".format(k_res[0], k_res[1], k_res[2], k_res[3]))
    
    
       
if __name__ == "__main__":
    # main(sys.argv)
    main()
