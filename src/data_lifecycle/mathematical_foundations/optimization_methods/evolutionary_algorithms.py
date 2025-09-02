from typing import Optional, Callable, Any, Dict, List
from general.base_classes.model_base import BaseModel
import numpy as np

class GenericEvolutionaryOptimizer(BaseModel):

    def __init__(self, dim: int, population_size: int=50, max_iterations: int=1000, selection_pressure: float=1.0, seed: Optional[int]=None, solution_callback: Optional[Callable[[Any], None]]=None):
        """
        Initialize the Generic Evolutionary Optimizer.
        
        Args:
            dim: Problem dimensionality (number of parameters to optimize).
            population_size: Number of individuals maintained during search.
            max_iterations: Stopping criterion based on iteration count.
            selection_pressure: Controls selective advantage of better solutions.
            seed: Seed for random number generator.
            solution_callback: Optional callback invoked with best solution each iteration.
        """
        super().__init__()
        self.dim = dim
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.selection_pressure = selection_pressure
        self.seed = seed
        self.solution_callback = solution_callback
        self._best_solution = None
        self._best_fitness = np.inf
        self._population = None
        self._fitness_history = []

    def fit(self, X: Any, y: Any=None, **kwargs) -> 'GenericEvolutionaryOptimizer':
        """
        Optimize an objective function using evolutionary principles.
        
        This method executes a generic evolutionary optimization loop,
        managing populations and applying variation operators.
        
        Args:
            X: Objective function to minimize. Should accept a 1D array-like input
               and return a scalar value representing the fitness.
            y: Not used, present for API compatibility.
            **kwargs: Additional keyword arguments passed to the optimizer.
            
        Returns:
            Self instance for method chaining.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        if not callable(X):
            raise TypeError('Objective function X must be callable')
        self._population = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness_values = np.array([X(individual) for individual in self._population])
        best_idx = np.argmin(fitness_values)
        self._best_fitness = fitness_values[best_idx]
        self._best_solution = self._population[best_idx].copy()
        self._fitness_history = [self._best_fitness]
        for iteration in range(self.max_iterations):
            selected_indices = self._tournament_selection(fitness_values)
            new_population = self._create_new_population(selected_indices)
            new_fitness_values = np.array([X(individual) for individual in new_population])
            self._population = new_population
            fitness_values = new_fitness_values
            current_best_idx = np.argmin(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            if current_best_fitness < self._best_fitness:
                self._best_fitness = current_best_fitness
                self._best_solution = self._population[current_best_idx].copy()
            self._fitness_history.append(self._best_fitness)
            if self.solution_callback is not None:
                try:
                    self.solution_callback(self._best_solution)
                except Exception:
                    pass
        return self

    def _tournament_selection(self, fitness_values: np.ndarray) -> np.ndarray:
        """Select individuals using tournament selection with selection pressure."""
        selected_indices = np.empty(self.population_size, dtype=int)
        adjusted_fitness = -fitness_values + np.max(fitness_values) + 1e-10
        for i in range(self.population_size):
            tournament_size = max(2, int(self.selection_pressure * 5))
            tournament_size = min(tournament_size, len(fitness_values))
            tournament_indices = np.random.choice(len(fitness_values), size=tournament_size, replace=False)
            tournament_fitness = adjusted_fitness[tournament_indices]
            selection_probs = np.power(tournament_fitness, max(1.0, self.selection_pressure))
            selection_probs = selection_probs / np.sum(selection_probs)
            winner_idx = np.random.choice(tournament_indices, p=selection_probs)
            selected_indices[i] = winner_idx
        return selected_indices

    def _create_new_population(self, selected_indices: np.ndarray) -> np.ndarray:
        """Create new population through crossover and mutation."""
        if self._population is None:
            self._population = np.random.uniform(-1, 1, (self.population_size, self.dim))
        new_population = np.empty_like(self._population)
        for i in range(self.population_size):
            parent1_idx = selected_indices[i]
            parent2_idx = selected_indices[np.random.randint(self.population_size)]
            alpha = np.random.uniform(-0.1, 1.1, self.dim)
            child = alpha * self._population[parent1_idx] + (1 - alpha) * self._population[parent2_idx]
            mutation_strength = 0.1 * (1 - i / self.max_iterations)
            child += np.random.normal(0, mutation_strength, self.dim)
            child = np.clip(child, -1, 1)
            new_population[i] = child
        return new_population

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Retrieve the best solution found during optimization.
        
        Args:
            X: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Best solution vector found during optimization.
        """
        if self._best_solution is None:
            return np.zeros(self.dim)
        return self._best_solution.copy()

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Return the fitness value of the best solution.
        
        Args:
            X: Placeholder argument for API compatibility.
            y: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Fitness value of the best solution.
        """
        if self._best_solution is None:
            return np.inf
        return self._best_fitness

class EvolutionaryStrategyOptimizer(BaseModel):
    """
    Evolutionary Strategy (ES) optimizer for continuous optimization problems.
    
    This optimizer uses principles of evolutionary computation where candidate
    solutions are represented as real-valued vectors. It employs mutation and
    selection operators to evolve solutions toward better fitness values.
    
    Attributes:
        dim (int): Dimensionality of the search space.
        population_size (int): Number of individuals in each generation.
        max_generations (int): Maximum number of generations allowed.
        mutation_strength (float): Initial standard deviation for mutations.
        recombination_ratio (float): Proportion of offspring created via recombination.
        survival_selection (str): Method for selecting survivors ('plus', 'comma').
        seed (Optional[int]): Random seed for reproducibility.
        solution_callback (Optional[Callable]): Callback function called after each generation.
    """

    def __init__(self, dim: int, population_size: int=50, max_generations: int=1000, mutation_strength: float=1.0, recombination_ratio: float=0.5, survival_selection: str='plus', seed: Optional[int]=None, solution_callback: Optional[Callable[[Any], None]]=None):
        """
        Initialize the Evolutionary Strategy optimizer.
        
        Args:
            dim: Problem dimensionality (number of parameters to optimize).
            population_size: Number of individuals in each generation.
            max_generations: Stopping criterion based on generation count.
            mutation_strength: Initial step size for Gaussian mutations.
            recombination_ratio: Fraction of offspring produced through recombination.
            survival_selection: Selection scheme for next generation ('plus' or 'comma').
            seed: Seed for random number generator.
            solution_callback: Optional callback invoked with best solution each generation.
        """
        super().__init__()
        if dim <= 0:
            raise ValueError('dim must be positive')
        if population_size <= 0:
            raise ValueError('population_size must be positive')
        if max_generations <= 0:
            raise ValueError('max_generations must be positive')
        if mutation_strength <= 0:
            raise ValueError('mutation_strength must be positive')
        if not 0 <= recombination_ratio <= 1:
            raise ValueError('recombination_ratio must be between 0 and 1')
        if survival_selection not in ['plus', 'comma']:
            raise ValueError("survival_selection must be either 'plus' or 'comma'")
        self.dim = dim
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_strength = mutation_strength
        self.recombination_ratio = recombination_ratio
        self.survival_selection = survival_selection
        self.seed = seed
        self.solution_callback = solution_callback
        self._best_solution = None
        self._best_fitness = np.inf
        self._offspring_size = population_size
        self._generation = 0
        self.is_fitted = False

    def fit(self, X: Any, y: Any=None, **kwargs) -> 'EvolutionaryStrategyOptimizer':
        """
        Optimize an objective function using Evolutionary Strategy.
        
        This method evolves a population of candidate solutions over multiple
        generations, applying mutation and recombination operators.
        
        Args:
            X: Objective function to minimize. Should accept a 1D array-like input
               and return a scalar value representing the fitness.
            y: Not used, present for API compatibility.
            **kwargs: Additional keyword arguments passed to the optimizer.
            
        Returns:
            Self instance for method chaining.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        if not callable(X):
            raise TypeError('Objective function X must be callable')
        parents = np.random.uniform(-1, 1, (self.population_size, self.dim))
        parent_fitnesses = np.array([X(individual) for individual in parents])
        best_idx = np.argmin(parent_fitnesses)
        self._best_fitness = parent_fitnesses[best_idx]
        self._best_solution = parents[best_idx].copy()
        sigma = self.mutation_strength
        self._generation = 0
        for generation in range(self.max_generations):
            self._generation = generation
            offspring = []
            offspring_fitnesses = []
            for _ in range(self._offspring_size):
                if np.random.rand() < self.recombination_ratio and self.population_size > 1:
                    parent_indices = np.random.choice(self.population_size, size=2, replace=False)
                    (parent1, parent2) = (parents[parent_indices[0]], parents[parent_indices[1]])
                    alpha = np.random.rand(self.dim)
                    child = alpha * parent1 + (1 - alpha) * parent2
                else:
                    parent_idx = np.random.randint(self.population_size)
                    child = parents[parent_idx].copy()
                child += np.random.normal(0, sigma, self.dim)
                child = np.clip(child, -1, 1)
                fitness = X(child)
                offspring.append(child)
                offspring_fitnesses.append(fitness)
                if fitness < self._best_fitness:
                    self._best_fitness = fitness
                    self._best_solution = child.copy()
            offspring = np.array(offspring)
            offspring_fitnesses = np.array(offspring_fitnesses)
            if self.survival_selection == 'plus':
                combined_pop = np.vstack([parents, offspring])
                combined_fitnesses = np.hstack([parent_fitnesses, offspring_fitnesses])
                survivor_indices = np.argsort(combined_fitnesses)[:self.population_size]
                parents = combined_pop[survivor_indices]
                parent_fitnesses = combined_fitnesses[survivor_indices]
            else:
                survivor_indices = np.argsort(offspring_fitnesses)[:self.population_size]
                parents = offspring[survivor_indices]
                parent_fitnesses = offspring_fitnesses[survivor_indices]
            if generation > 0 and generation % 10 == 0:
                sigma *= 0.99
            if self.solution_callback is not None:
                try:
                    self.solution_callback(self._best_solution)
                except Exception:
                    pass
        self._generation = self.max_generations
        self.is_fitted = True
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Retrieve the best solution found during optimization.
        
        Args:
            X: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Best solution vector found during optimization.
        """
        if not self.is_fitted:
            return np.zeros(self.dim)
        return self._best_solution.copy()

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Return the fitness value of the best solution.
        
        Args:
            X: Placeholder argument for API compatibility.
            y: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Fitness value of the best solution.
        """
        if not self.is_fitted:
            return np.inf
        return self._best_fitness

class GeneticAlgorithmOptimizer(BaseModel):
    """
    Genetic Algorithm (GA) optimizer for solving optimization problems.
    
    This optimizer uses principles of natural selection and genetics to evolve
    solutions over generations. It maintains a population of candidate solutions
    that undergo selection, crossover, and mutation operations to improve fitness.
    
    Attributes:
        dim (int): Dimensionality of the search space.
        population_size (int): Number of individuals in each generation.
        max_generations (int): Maximum number of generations allowed.
        crossover_rate (float): Probability of crossover between parents.
        mutation_rate (float): Probability of mutation for each gene.
        selection_method (str): Method for selecting parents ('tournament', 'roulette').
        elitism_count (int): Number of best individuals preserved in each generation.
        seed (Optional[int]): Random seed for reproducibility.
        solution_callback (Optional[Callable]): Callback function called after each generation.
    """

    def __init__(self, dim: int, population_size: int=50, max_generations: int=1000, crossover_rate: float=0.8, mutation_rate: float=0.1, selection_method: str='tournament', elitism_count: int=1, seed: Optional[int]=None, solution_callback: Optional[Callable[[Any], None]]=None):
        """
        Initialize the Genetic Algorithm optimizer.
        
        Args:
            dim: Problem dimensionality (number of parameters to optimize).
            population_size: Number of individuals in each generation.
            max_generations: Stopping criterion based on generation count.
            crossover_rate: Likelihood of combining two parent solutions.
            mutation_rate: Likelihood of altering genes in offspring.
            selection_method: Technique for choosing parents for reproduction.
            elitism_count: Number of top individuals carried over to next generation.
            seed: Seed for random number generator.
            solution_callback: Optional callback invoked with best solution each generation.
        """
        super().__init__()
        self.dim = dim
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.elitism_count = elitism_count
        self.seed = seed
        self.solution_callback = solution_callback
        self._best_solution = None
        self._best_fitness = np.inf
        self._population = None
        self._fitness_values = None

    def fit(self, X: Any, y: Any=None, **kwargs) -> 'GeneticAlgorithmOptimizer':
        """
        Optimize an objective function using Genetic Algorithm.
        
        This method evolves a population of candidate solutions over multiple
        generations, applying genetic operators to explore the search space.
        
        Args:
            X: Objective function to minimize. Should accept a 1D array-like input
               and return a scalar value representing the fitness.
            y: Not used, present for API compatibility.
            **kwargs: Additional keyword arguments passed to the optimizer.
            
        Returns:
            Self instance for method chaining.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        if not callable(X):
            raise TypeError('Objective function X must be callable')
        self._population = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self._fitness_values = np.array([X(individual) for individual in self._population])
        best_idx = np.argmin(self._fitness_values)
        self._best_fitness = self._fitness_values[best_idx]
        self._best_solution = self._population[best_idx].copy()
        for generation in range(self.max_generations):
            elite_indices = np.argsort(self._fitness_values)[:self.elitism_count]
            new_population = self._population[elite_indices].copy()
            while len(new_population) < self.population_size:
                if self.selection_method == 'tournament':
                    parent1_idx = self._tournament_selection()
                    parent2_idx = self._tournament_selection()
                elif self.selection_method == 'roulette':
                    parent1_idx = self._roulette_wheel_selection()
                    parent2_idx = self._roulette_wheel_selection()
                else:
                    raise ValueError(f'Unknown selection method: {self.selection_method}')
                if np.random.random() < self.crossover_rate:
                    (child1, child2) = self._crossover(parent1_idx, parent2_idx)
                else:
                    child1 = self._population[parent1_idx].copy()
                    child2 = self._population[parent2_idx].copy()
                self._mutate(child1)
                self._mutate(child2)
                new_population = np.vstack([new_population, child1.reshape(1, -1)])
                if len(new_population) < self.population_size:
                    new_population = np.vstack([new_population, child2.reshape(1, -1)])
            self._population = new_population[:self.population_size]
            self._fitness_values = np.array([X(individual) for individual in self._population])
            current_best_idx = np.argmin(self._fitness_values)
            current_best_fitness = self._fitness_values[current_best_idx]
            if current_best_fitness < self._best_fitness:
                self._best_fitness = current_best_fitness
                self._best_solution = self._population[current_best_idx].copy()
            if self.solution_callback is not None:
                try:
                    self.solution_callback(self._best_solution)
                except Exception:
                    pass
        self.is_fitted = True
        return self

    def _tournament_selection(self) -> int:
        """Perform tournament selection and return index of selected individual."""
        tournament_size = max(2, self.population_size // 5)
        tournament_indices = np.random.choice(self.population_size, size=tournament_size, replace=False)
        tournament_fitness = self._fitness_values[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return winner_idx

    def _roulette_wheel_selection(self) -> int:
        """Perform roulette wheel selection and return index of selected individual."""
        inverted_fitness = 1.0 / (self._fitness_values + 1e-10)
        total_fitness = np.sum(inverted_fitness)
        probabilities = inverted_fitness / total_fitness
        selected_index = np.random.choice(self.population_size, p=probabilities)
        return selected_index

    def _crossover(self, parent1_idx: int, parent2_idx: int) -> tuple:
        """Perform uniform crossover between two parents."""
        parent1 = self._population[parent1_idx]
        parent2 = self._population[parent2_idx]
        mask = np.random.random(self.dim) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return (child1, child2)

    def _mutate(self, individual: np.ndarray) -> None:
        """Apply Gaussian mutation to an individual."""
        mutation_mask = np.random.random(self.dim) < self.mutation_rate
        mutation_noise = np.random.normal(0, 0.1, self.dim)
        individual[mutation_mask] += mutation_noise[mutation_mask]
        np.clip(individual, -1, 1, out=individual)

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Retrieve the best solution found during optimization.
        
        Args:
            X: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Best solution vector found during optimization.
        """
        if not self.is_fitted or self._best_solution is None:
            return np.zeros(self.dim)
        return self._best_solution.copy()

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Return the fitness value of the best solution.
        
        Args:
            X: Placeholder argument for API compatibility.
            y: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Fitness value of the best solution.
        """
        if not self.is_fitted or self._best_solution is None:
            return np.inf
        return self._best_fitness

class ParticleSwarmOptimizer(BaseModel):
    """
    Particle Swarm Optimization (PSO) algorithm implementation.
    
    This optimizer simulates social behavior of swarms to solve optimization problems.
    Particles move through the search space, adjusting their positions based on their
    own best-known position and the swarm's global best position.
    
    Attributes:
        dim (int): Dimensionality of the search space.
        num_particles (int): Number of particles in the swarm.
        max_iterations (int): Maximum number of iterations allowed.
        inertia_weight (float): Controls balance between exploration and exploitation.
        cognitive_coefficient (float): Weight for particle's personal best influence.
        social_coefficient (float): Weight for swarm's global best influence.
        seed (Optional[int]): Random seed for reproducibility.
        solution_callback (Optional[Callable]): Callback function called after each iteration.
    """

    def __init__(self, dim: int, num_particles: int=30, max_iterations: int=1000, inertia_weight: float=0.729, cognitive_coefficient: float=1.494, social_coefficient: float=1.494, seed: Optional[int]=None, solution_callback: Optional[Callable[[Any], None]]=None):
        """
        Initialize the PSO optimizer.
        
        Args:
            dim: Problem dimensionality (number of parameters to optimize).
            num_particles: Number of particles in the swarm.
            max_iterations: Stopping criterion based on iteration count.
            inertia_weight: Velocity retention factor between iterations.
            cognitive_coefficient: Influence of particle's personal best position.
            social_coefficient: Influence of swarm's global best position.
            seed: Seed for random number generator.
            solution_callback: Optional callback invoked with best solution each iteration.
        """
        super().__init__()
        self.dim = dim
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.seed = seed
        self.solution_callback = solution_callback
        self._best_solution = None
        self._best_fitness = np.inf
        self._positions = None
        self._velocities = None
        self._personal_bests = None
        self._personal_best_fitnesses = None
        self._global_best_position = None
        self._global_best_fitness = np.inf

    def fit(self, X: Any, y: Any=None, **kwargs) -> 'ParticleSwarmOptimizer':
        """
        Optimize an objective function using Particle Swarm Optimization.
        
        This method initializes a swarm of particles and evolves them over iterations
        by updating velocities and positions according to PSO update equations.
        
        Args:
            X: Objective function to minimize. Should accept a 1D array-like input
               and return a scalar value representing the fitness.
            y: Not used, present for API compatibility.
            **kwargs: Additional keyword arguments passed to the optimizer.
            
        Returns:
            Self instance for method chaining.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        if not callable(X):
            raise TypeError('Objective function X must be callable')
        self._positions = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self._velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))
        self._personal_bests = self._positions.copy()
        self._personal_best_fitnesses = np.full(self.num_particles, np.inf)
        fitness_values = np.array([X(position) for position in self._positions])
        self._personal_best_fitnesses = fitness_values.copy()
        best_idx = np.argmin(fitness_values)
        self._global_best_fitness = fitness_values[best_idx]
        self._global_best_position = self._positions[best_idx].copy()
        self._best_solution = self._global_best_position.copy()
        self._best_fitness = self._global_best_fitness
        for iteration in range(self.max_iterations):
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self._velocities[i] = self.inertia_weight * self._velocities[i] + self.cognitive_coefficient * r1 * (self._personal_bests[i] - self._positions[i]) + self.social_coefficient * r2 * (self._global_best_position - self._positions[i])
                self._positions[i] += self._velocities[i]
                self._positions[i] = np.clip(self._positions[i], -1, 1)
                fitness = X(self._positions[i])
                if fitness < self._personal_best_fitnesses[i]:
                    self._personal_bests[i] = self._positions[i].copy()
                    self._personal_best_fitnesses[i] = fitness
                    if fitness < self._global_best_fitness:
                        self._global_best_fitness = fitness
                        self._global_best_position = self._positions[i].copy()
                        self._best_solution = self._global_best_position.copy()
                        self._best_fitness = self._global_best_fitness
            if self.solution_callback is not None:
                try:
                    self.solution_callback(self._best_solution)
                except Exception:
                    pass
        self.is_fitted = True
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Retrieve the best solution found during optimization.
        
        Args:
            X: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Best solution vector found during optimization.
        """
        if not self.is_fitted:
            return np.zeros(self.dim)
        return self._best_solution.copy()

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Return the fitness value of the best solution.
        
        Args:
            X: Placeholder argument for API compatibility.
            y: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Fitness value of the best solution.
        """
        if not self.is_fitted:
            return np.inf
        return self._best_fitness


# ...(code omitted)...


class CovarianceMatrixAdaptationEvolutionStrategy(BaseModel):

    def __init__(self, dim: int, population_size: Optional[int]=None, initial_step_size: float=1.0, max_iterations: int=1000, seed: Optional[int]=None, solution_callback: Optional[Callable[[Any], None]]=None):
        """
        Initialize the CMA-ES optimizer.
        
        Args:
            dim: Problem dimensionality (number of parameters to optimize).
            population_size: Population size; defaults to 4 + int(3 * log(dim)).
            initial_step_size: Starting global step size for mutations.
            max_iterations: Stopping criterion based on iteration count.
            seed: Seed for random number generator.
            solution_callback: Optional callback invoked with best solution each iteration.
        """
        super().__init__()
        if dim <= 0:
            raise ValueError('dim must be positive')
        if population_size is not None and population_size <= 0:
            raise ValueError('population_size must be positive')
        if initial_step_size <= 0:
            raise ValueError('initial_step_size must be positive')
        if max_iterations <= 0:
            raise ValueError('max_iterations must be positive')
        self.dim = dim
        self.population_size = population_size or 4 + int(3 * np.log(dim))
        self.initial_step_size = initial_step_size
        self.max_iterations = max_iterations
        self.seed = seed
        self.solution_callback = solution_callback
        self.is_fitted = False

    def fit(self, X: Any, y: Any=None, **kwargs) -> 'CovarianceMatrixAdaptationEvolutionStrategy':
        """
        Optimize an objective function using CMA-ES.
        
        This method runs the full optimization loop, generating populations,
        evaluating fitness, and adapting distribution parameters.
        
        Args:
            X: Objective function to minimize. Should accept a 1D array-like input
               and return a scalar value representing the fitness.
            y: Not used, present for API compatibility.
            **kwargs: Additional keyword arguments passed to the optimizer.
            
        Returns:
            Self instance for method chaining.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        if not callable(X):
            raise TypeError('Objective function X must be callable')
        mu = self.population_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        cc = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + cs
        chiN = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        x_mean = np.random.randn(self.dim)
        sigma = self.initial_step_size
        C = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        self._best_solution = x_mean.copy()
        self._best_fitness = X(x_mean)
        for iteration in range(self.max_iterations):
            arz = np.random.randn(self.population_size, self.dim)
            arx = np.array([x_mean + sigma * np.dot(arz[i], np.linalg.cholesky(C).T) for i in range(self.population_size)])
            fitness = np.array([X(individual) for individual in arx])
            indices = np.argsort(fitness)
            arx_sorted = arx[indices]
            arz_sorted = arz[indices]
            fitness_sorted = fitness[indices]
            if fitness_sorted[0] < self._best_fitness:
                self._best_fitness = fitness_sorted[0]
                self._best_solution = arx_sorted[0].copy()
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * arx_sorted[:mu], axis=0)
            ps_tmp = (x_mean - x_old) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * ps_tmp
            hsig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1))) / chiN < 1.4 + 2 / (self.dim + 1)
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * ps_tmp
            artmp = 1 / sigma * (arx_sorted[:mu] - x_old)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * C) + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * (artmp[:, :, np.newaxis] * artmp[:, np.newaxis, :]), axis=0)
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(p_sigma) / chiN - 1))
            if self.solution_callback is not None:
                try:
                    self.solution_callback(self._best_solution)
                except Exception:
                    pass
        self.is_fitted = True
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Retrieve the best solution found during optimization.
        
        Args:
            X: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Best solution vector found during optimization.
        """
        if not self.is_fitted:
            return np.zeros(self.dim)
        return self._best_solution.copy()

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Return the fitness value of the best solution.
        
        Args:
            X: Placeholder argument for API compatibility.
            y: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Fitness value of the best solution.
        """
        if not self.is_fitted:
            return np.inf
        return self._best_fitness

class ImplementedCovarianceMatrixAdaptationEvolutionStrategy(BaseModel):

    def __init__(self, dim: int, population_size: Optional[int]=None, initial_step_size: float=1.0, max_iterations: int=1000, convergence_tolerance: float=1e-12, seed: Optional[int]=None, solution_callback: Optional[Callable[[Any], None]]=None):
        """
        Initialize the implemented CMA-ES optimizer.
        
        Args:
            dim: Problem dimensionality (number of parameters to optimize).
            population_size: Population size; defaults to 4 + int(3 * log(dim)).
            initial_step_size: Starting global step size for mutations.
            max_iterations: Stopping criterion based on iteration count.
            convergence_tolerance: Minimum change threshold for termination.
            seed: Seed for random number generator.
            solution_callback: Optional callback invoked with best solution each iteration.
        """
        super().__init__()
        if dim <= 0:
            raise ValueError('dim must be positive')
        if population_size is not None and population_size <= 0:
            raise ValueError('population_size must be positive')
        if initial_step_size <= 0:
            raise ValueError('initial_step_size must be positive')
        if max_iterations <= 0:
            raise ValueError('max_iterations must be positive')
        if convergence_tolerance <= 0:
            raise ValueError('convergence_tolerance must be positive')
        self.dim = dim
        self.population_size = population_size or 4 + int(3 * np.log(dim))
        self.initial_step_size = initial_step_size
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.seed = seed
        self.solution_callback = solution_callback
        self._best_solution = None
        self._best_fitness = np.inf
        self.is_fitted = False

    def fit(self, X: Any, y: Any=None, **kwargs) -> 'ImplementedCovarianceMatrixAdaptationEvolutionStrategy':
        """
        Optimize an objective function using the implemented CMA-ES algorithm.
        
        This method runs the complete CMA-ES optimization loop with full
        covariance matrix and step-size adaptation mechanisms.
        
        Args:
            X: Objective function to minimize. Should accept a 1D array-like input
               and return a scalar value representing the fitness.
            y: Not used, present for API compatibility.
            **kwargs: Additional keyword arguments passed to the optimizer.
            
        Returns:
            Self instance for method chaining.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        if not callable(X):
            raise TypeError('Objective function X must be callable')
        mu = self.population_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + cs
        cc = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
        chiN = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        x_mean = np.zeros(self.dim)
        sigma = self.initial_step_size
        C = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        self._best_solution = x_mean.copy()
        self._best_fitness = X(x_mean)
        for iteration in range(self.max_iterations):
            arz = np.random.randn(self.population_size, self.dim)
            arx = np.array([x_mean + sigma * np.dot(arz[i], np.linalg.cholesky(C).T) for i in range(self.population_size)])
            fitness = np.array([X(individual) for individual in arx])
            indices = np.argsort(fitness)
            arx_sorted = arx[indices]
            arz_sorted = arz[indices]
            fitness_sorted = fitness[indices]
            if fitness_sorted[0] < self._best_fitness:
                self._best_fitness = fitness_sorted[0]
                self._best_solution = arx_sorted[0].copy()
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * arx_sorted[:mu], axis=0)
            ps_tmp = (x_mean - x_old) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * ps_tmp
            hsig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1))) / chiN < 1.4 + 2 / (self.dim + 1)
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * ps_tmp
            artmp = 1 / sigma * (arx_sorted[:mu] - x_old)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * C) + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * (artmp[:, :, np.newaxis] * artmp[:, np.newaxis, :]), axis=0)
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
            if np.linalg.norm(x_mean - x_old) < self.convergence_tolerance:
                break
            if self.solution_callback is not None:
                try:
                    self.solution_callback(self._best_solution)
                except Exception:
                    pass
        self.is_fitted = True
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Retrieve the best solution found during optimization.
        
        Args:
            X: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Best solution vector found during optimization.
        """
        if not self.is_fitted or self._best_solution is None:
            return np.zeros(self.dim)
        return self._best_solution.copy()

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Return the fitness value of the best solution.
        
        Args:
            X: Placeholder argument for API compatibility.
            y: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Fitness value of the best solution.
        """
        if not self.is_fitted:
            return np.inf
        return self._best_fitness

class CMAESWithLearningRatesOptimizer(BaseModel):

    def __init__(self, dim: int, population_size: int=None, initial_step_size: float=1.0, mean_lr: float=1.0, cov_lr: float=1.0, max_iterations: int=1000, seed: Optional[int]=None, solution_callback: Optional[Callable[[Any], None]]=None):
        """
        Initialize the CMA-ES optimizer with learning rates.
        
        Args:
            dim: Problem dimensionality (number of parameters to optimize).
            population_size: Population size; defaults to 4 + int(3 * log(dim)).
            initial_step_size: Starting global step size for mutations.
            mean_lr: Learning rate for mean vector adaptation.
            cov_lr: Learning rate for covariance matrix adaptation.
            max_iterations: Stopping criterion based on iteration count.
            seed: Seed for random number generator.
            solution_callback: Optional callback invoked with best solution each iteration.
        """
        super().__init__()
        if dim <= 0:
            raise ValueError('dim must be positive')
        if population_size is not None and population_size <= 0:
            raise ValueError('population_size must be positive')
        if initial_step_size <= 0:
            raise ValueError('initial_step_size must be positive')
        if mean_lr <= 0:
            raise ValueError('mean_lr must be positive')
        if cov_lr <= 0:
            raise ValueError('cov_lr must be positive')
        if max_iterations <= 0:
            raise ValueError('max_iterations must be positive')
        self.dim = dim
        self.population_size = population_size or 4 + int(3 * np.log(dim))
        self.initial_step_size = initial_step_size
        self.mean_lr = mean_lr
        self.cov_lr = cov_lr
        self.max_iterations = max_iterations
        self.seed = seed
        self.solution_callback = solution_callback
        self.is_fitted = False

    def fit(self, X: Any, y: Any=None, **kwargs) -> 'CMAESWithLearningRatesOptimizer':
        """
        Optimize an objective function using CMA-ES with learning rates.
        
        This method runs the full optimization loop, generating populations,
        evaluating fitness, adapting distribution parameters, and applying
        configured learning rates throughout the process.
        
        Args:
            X: Objective function to minimize. Should accept a 1D array-like input
               and return a scalar value representing the fitness.
            y: Not used, present for API compatibility.
            **kwargs: Additional keyword arguments passed to the optimizer.
            
        Returns:
            Self instance for method chaining.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        sigma = self.initial_step_size
        x_mean = np.random.randn(self.dim)
        C = np.eye(self.dim)
        mu = self.population_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        cc = 4 / (self.dim + 4)
        cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c1 = self.cov_lr * 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, self.cov_lr * mu_eff / (2 * (self.dim + np.sqrt(mu_eff)) ** 2))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + cs
        p_c = np.zeros(self.dim)
        p_sigma = np.zeros(self.dim)
        chiN = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        self._best_solution = x_mean.copy()
        self._best_fitness = X(x_mean)
        for iteration in range(self.max_iterations):
            arz = np.random.randn(self.population_size, self.dim)
            try:
                chol_C = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += np.eye(self.dim) * 1e-10
                chol_C = np.linalg.cholesky(C)
            arx = np.array([x_mean + sigma * np.dot(arz[i], chol_C.T) for i in range(self.population_size)])
            fitness = np.array([X(individual) for individual in arx])
            indices = np.argsort(fitness)
            arx_sorted = arx[indices]
            fitness_sorted = fitness[indices]
            if fitness_sorted[0] < self._best_fitness:
                self._best_fitness = fitness_sorted[0]
                self._best_solution = arx_sorted[0].copy()
            x_old = x_mean.copy()
            x_mean = x_mean + self.mean_lr * np.sum(weights[:, np.newaxis] * (arx_sorted[:mu] - x_mean), axis=0)
            ps_tmp = (x_mean - x_old) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * ps_tmp
            hsig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1))) / chiN < 1.4 + 2 / (self.dim + 1)
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * ps_tmp
            artmp = 1 / sigma * (arx_sorted[:mu] - x_old)
            C_factor = np.sum(weights[:, np.newaxis, np.newaxis] * (artmp[:, :, np.newaxis] * artmp[:, np.newaxis, :]), axis=0)
            C = C + self.cov_lr * (c1 * (np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (C_factor - C))
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(p_sigma) / chiN - 1))
            if self.solution_callback is not None:
                try:
                    self.solution_callback(self._best_solution)
                except Exception:
                    pass
        self.is_fitted = True
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Retrieve the best solution found during optimization.
        
        Args:
            X: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Best solution vector found during optimization.
        """
        if not self.is_fitted or self._best_solution is None:
            return np.zeros(self.dim)
        return self._best_solution.copy()

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Return the fitness value of the best solution.
        
        Args:
            X: Placeholder argument for API compatibility.
            y: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Fitness value of the best solution.
        """
        if not self.is_fitted:
            return np.inf
        return self._best_fitness

class GeneticAlgorithmForOptimization(BaseModel):
    """
    Specialized Genetic Algorithm implementation for optimization tasks.
    
    This optimizer implements a canonical genetic algorithm with configurable
    genetic operators. It is specifically tailored for direct optimization
    applications rather than hyperparameter tuning.
    
    Attributes:
        dim (int): Dimensionality of the search space.
        population_size (int): Number of individuals in each generation.
        max_generations (int): Maximum number of generations allowed.
        crossover_probability (float): Probability of crossover between parents.
        mutation_probability (float): Probability of mutation for each gene.
        selection_type (str): Type of selection mechanism ('tournament', 'roulette').
        elitism_size (int): Number of elite individuals preserved each generation.
        seed (Optional[int]): Random seed for reproducibility.
        solution_callback (Optional[Callable]): Callback function called after each generation.
    """

    def __init__(self, dim: int, population_size: int=50, max_generations: int=1000, crossover_probability: float=0.8, mutation_probability: float=0.1, selection_type: str='tournament', elitism_size: int=1, seed: Optional[int]=None, solution_callback: Optional[Callable[[Any], None]]=None):
        """
        Initialize the Genetic Algorithm optimizer.
        
        Args:
            dim: Problem dimensionality (number of parameters to optimize).
            population_size: Number of individuals in each generation.
            max_generations: Stopping criterion based on generation count.
            crossover_probability: Likelihood of combining two parent solutions.
            mutation_probability: Likelihood of altering genes in offspring.
            selection_type: Mechanism for choosing parents for reproduction.
            elitism_size: Number of top individuals carried over to next generation.
            seed: Seed for random number generator.
            solution_callback: Optional callback invoked with best solution each generation.
        """
        super().__init__()
        self.dim = dim
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.selection_type = selection_type
        self.elitism_size = elitism_size
        self.seed = seed
        self.solution_callback = solution_callback
        self._best_solution = None
        self._best_fitness = np.inf
        self._population = None
        self._fitness_values = None

    def fit(self, X: Any, y: Any=None, **kwargs) -> 'GeneticAlgorithmForOptimization':
        """
        Optimize an objective function using Genetic Algorithm principles.
        
        This method evolves a population of candidate solutions over multiple
        generations, applying genetic operators to explore the search space.
        
        Args:
            X: Objective function to minimize. Should accept a 1D array-like input
               and return a scalar value representing the fitness.
            y: Not used, present for API compatibility.
            **kwargs: Additional keyword arguments passed to the optimizer.
            
        Returns:
            Self instance for method chaining.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        if not callable(X):
            raise TypeError('Objective function X must be callable')
        self._population = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self._fitness_values = np.array([X(individual) for individual in self._population])
        best_idx = np.argmin(self._fitness_values)
        self._best_fitness = self._fitness_values[best_idx]
        self._best_solution = self._population[best_idx].copy()
        for generation in range(self.max_generations):
            elite_indices = np.argsort(self._fitness_values)[:self.elitism_size]
            new_population = self._population[elite_indices].copy()
            while len(new_population) < self.population_size:
                if self.selection_type == 'tournament':
                    parent1_idx = self._tournament_selection()
                    parent2_idx = self._tournament_selection()
                elif self.selection_type == 'roulette':
                    parent1_idx = self._roulette_wheel_selection()
                    parent2_idx = self._roulette_wheel_selection()
                else:
                    raise ValueError(f'Unknown selection method: {self.selection_type}')
                if np.random.random() < self.crossover_probability:
                    (child1, child2) = self._crossover(parent1_idx, parent2_idx)
                else:
                    child1 = self._population[parent1_idx].copy()
                    child2 = self._population[parent2_idx].copy()
                self._mutate(child1)
                self._mutate(child2)
                new_population = np.vstack([new_population, child1.reshape(1, -1)])
                if len(new_population) < self.population_size:
                    new_population = np.vstack([new_population, child2.reshape(1, -1)])
            self._population = new_population[:self.population_size]
            self._fitness_values = np.array([X(individual) for individual in self._population])
            current_best_idx = np.argmin(self._fitness_values)
            current_best_fitness = self._fitness_values[current_best_idx]
            if current_best_fitness < self._best_fitness:
                self._best_fitness = current_best_fitness
                self._best_solution = self._population[current_best_idx].copy()
            if self.solution_callback is not None:
                try:
                    self.solution_callback(self._best_solution)
                except Exception:
                    pass
        self.is_fitted = True
        return self

    def _tournament_selection(self) -> int:
        """Perform tournament selection and return index of selected individual."""
        tournament_size = max(2, self.population_size // 5)
        tournament_indices = np.random.choice(self.population_size, size=tournament_size, replace=False)
        tournament_fitness = self._fitness_values[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return winner_idx

    def _roulette_wheel_selection(self) -> int:
        """Perform roulette wheel selection and return index of selected individual."""
        inverted_fitness = 1.0 / (self._fitness_values + 1e-10)
        total_fitness = np.sum(inverted_fitness)
        probabilities = inverted_fitness / total_fitness
        selected_index = np.random.choice(self.population_size, p=probabilities)
        return selected_index

    def _crossover(self, parent1_idx: int, parent2_idx: int) -> tuple:
        """Perform uniform crossover between two parents."""
        parent1 = self._population[parent1_idx]
        parent2 = self._population[parent2_idx]
        mask = np.random.random(self.dim) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return (child1, child2)

    def _mutate(self, individual: np.ndarray) -> None:
        """Apply Gaussian mutation to an individual."""
        mutation_mask = np.random.random(self.dim) < self.mutation_probability
        mutation_noise = np.random.normal(0, 0.1, self.dim)
        individual[mutation_mask] += mutation_noise[mutation_mask]
        np.clip(individual, -1, 1, out=individual)

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Retrieve the best solution found during optimization.
        
        Args:
            X: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Best solution vector found during optimization.
        """
        if not self.is_fitted or self._best_solution is None:
            return np.zeros(self.dim)
        return self._best_solution.copy()

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Return the fitness value of the best solution.
        
        Args:
            X: Placeholder argument for API compatibility.
            y: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Fitness value of the best solution.
        """
        if not self.is_fitted or self._best_solution is None:
            return np.inf
        return self._best_fitness

class EvolutionaryStrategyOptimization(BaseModel):

    def __init__(self, dim: int, population_size: int=50, max_generations: int=1000, initial_sigma: float=1.0, recombination_fraction: float=0.5, survival_method: str='plus', seed: Optional[int]=None, solution_callback: Optional[Callable[[Any], None]]=None):
        """
        Initialize the Evolutionary Strategy optimizer.
        
        Args:
            dim: Problem dimensionality (number of parameters to optimize).
            population_size: Number of individuals in each generation.
            max_generations: Stopping criterion based on generation count.
            initial_sigma: Starting mutation step size for all individuals.
            recombination_fraction: Ratio of offspring created via recombination.
            survival_method: Method for selecting survivors for next generation.
            seed: Seed for random number generator.
            solution_callback: Optional callback invoked with best solution each generation.
        """
        super().__init__()
        if dim <= 0:
            raise ValueError('dim must be positive')
        if population_size <= 0:
            raise ValueError('population_size must be positive')
        if max_generations <= 0:
            raise ValueError('max_generations must be positive')
        if initial_sigma <= 0:
            raise ValueError('initial_sigma must be positive')
        if not 0 <= recombination_fraction <= 1:
            raise ValueError('recombination_fraction must be between 0 and 1')
        if survival_method not in ['plus', 'comma']:
            raise ValueError("survival_method must be either 'plus' or 'comma'")
        self.dim = dim
        self.population_size = population_size
        self.max_generations = max_generations
        self.initial_sigma = initial_sigma
        self.recombination_fraction = recombination_fraction
        self.survival_method = survival_method
        self.seed = seed
        self.solution_callback = solution_callback
        self._best_solution = None
        self._best_fitness = np.inf
        self.is_fitted = False

    def fit(self, X: Any, y: Any=None, **kwargs) -> 'EvolutionaryStrategyOptimization':
        """
        Optimize an objective function using Evolutionary Strategy principles.
        
        This method evolves a population with self-adaptive mutation parameters
        to efficiently navigate continuous search spaces.
        
        Args:
            X: Objective function to minimize. Should accept a 1D array-like input
               and return a scalar value representing the fitness.
            y: Not used, present for API compatibility.
            **kwargs: Additional keyword arguments passed to the optimizer.
            
        Returns:
            Self instance for method chaining.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        if not callable(X):
            raise TypeError('Objective function X must be callable')
        population = []
        for _ in range(self.population_size):
            individual = {'solution': np.random.uniform(-5, 5, self.dim), 'sigma': np.full(self.dim, self.initial_sigma)}
            population.append(individual)
        for individual in population:
            individual['fitness'] = X(individual['solution'])
        for generation in range(self.max_generations):
            offspring = []
            num_recombined = int(self.recombination_fraction * self.population_size)
            num_mutated = self.population_size - num_recombined
            for _ in range(num_recombined // 2):
                parent1_idx = self._select_parent(population)
                parent2_idx = self._select_parent(population)
                (child1, child2) = self._recombine(population[parent1_idx], population[parent2_idx])
                self._mutate_individual(child1)
                self._mutate_individual(child2)
                child1['fitness'] = X(child1['solution'])
                child2['fitness'] = X(child2['solution'])
                offspring.extend([child1, child2])
            for _ in range(num_mutated):
                parent_idx = self._select_parent(population)
                parent = population[parent_idx]
                child = {'solution': parent['solution'].copy(), 'sigma': parent['sigma'].copy()}
                self._mutate_individual(child)
                child['fitness'] = X(child['solution'])
                offspring.append(child)
            if self.survival_method == 'plus':
                combined = population + offspring
                combined.sort(key=lambda ind: ind['fitness'])
                population = combined[:self.population_size]
            else:
                offspring.sort(key=lambda ind: ind['fitness'])
                population = offspring[:self.population_size]
            current_best = min(population, key=lambda ind: ind['fitness'])
            if current_best['fitness'] < self._best_fitness:
                self._best_fitness = current_best['fitness']
                self._best_solution = current_best['solution'].copy()
            if self.solution_callback is not None:
                try:
                    self.solution_callback(self._best_solution)
                except Exception:
                    pass
        self.is_fitted = True
        return self

    def _select_parent(self, population: List[Dict]) -> int:
        """Select a parent using tournament selection."""
        tournament_size = min(3, len(population))
        indices = np.random.choice(len(population), size=tournament_size, replace=False)
        fitnesses = np.array([individual['fitness'] for individual in [population[i] for i in indices]])
        return int(indices[np.argmin(fitnesses)])

    def _recombine(self, parent1: Dict[str, np.ndarray], parent2: Dict[str, np.ndarray]) -> tuple:
        """Perform intermediate recombination between two parents."""
        alpha_sol = np.random.uniform(0, 1, self.dim)
        child1_solution = alpha_sol * parent1['solution'] + (1 - alpha_sol) * parent2['solution']
        child2_solution = (1 - alpha_sol) * parent1['solution'] + alpha_sol * parent2['solution']
        alpha_sigma = np.random.uniform(0, 1, self.dim)
        child1_sigma = alpha_sigma * parent1['sigma'] + (1 - alpha_sigma) * parent2['sigma']
        child2_sigma = (1 - alpha_sigma) * parent1['sigma'] + alpha_sigma * parent2['sigma']
        child1 = {'solution': child1_solution, 'sigma': child1_sigma}
        child2 = {'solution': child2_solution, 'sigma': child2_sigma}
        return (child1, child2)

    def _mutate_individual(self, individual: Dict) -> None:
        """Apply self-adaptive mutation to an individual."""
        tau = 1.0 / np.sqrt(2 * np.sqrt(self.dim))
        tau_prime = 1.0 / np.sqrt(2 * self.dim)
        epsilon = 1e-15
        eta = np.random.normal(0, 1)
        individual['sigma'] *= np.exp(tau_prime * eta + tau * np.random.normal(0, 1, self.dim))
        individual['sigma'] = np.maximum(individual['sigma'], epsilon)
        individual['solution'] += individual['sigma'] * np.random.normal(0, 1, self.dim)
        np.clip(individual['solution'], -100, 100, out=individual['solution'])

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Retrieve the best solution found during optimization.
        
        Args:
            X: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Best solution vector found during optimization.
        """
        if not self.is_fitted or self._best_solution is None:
            return np.zeros(self.dim)
        return self._best_solution.copy()

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Return the fitness value of the best solution.
        
        Args:
            X: Placeholder argument for API compatibility.
            y: Placeholder argument for API compatibility.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Fitness value of the best solution.
        """
        if not self.is_fitted or self._best_solution is None:
            return np.inf
        return self._best_fitness