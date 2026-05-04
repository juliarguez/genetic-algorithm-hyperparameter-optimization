import random
from deap import base, creator, tools, algorithms
from model import evaluate_model


# 1. Fitness function (maximization)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# 2. Toolbox setup

toolbox = base.Toolbox()

# Hyperparameter ranges
toolbox.register("n_estimators", random.randint, 10, 200)
toolbox.register("max_depth", random.randint, 1, 20)
toolbox.register("min_samples_split", random.randint, 2, 10)


# Individual = [n_estimators, max_depth, min_samples_split]
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.n_estimators, toolbox.max_depth, toolbox.min_samples_split),
    n=1
)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# 3. Evaluation function

def eval_individual(individual):
    n_estimators, max_depth, min_samples_split = individual

    score = evaluate_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )

    return (score,)  # DEAP expects tuple


toolbox.register("evaluate", eval_individual)


# 4. Genetic operators

toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", tools.mutUniformInt,
                 low=[10, 1, 2],
                 up=[200, 20, 10],
                 indpb=0.2)

toolbox.register("select", tools.selTournament, tournsize=3)


# 5. Main GA function

def run_genetic_algorithm(pop_size=10, n_gen=5):

    pop = toolbox.population(n=pop_size)

    print("\n=== GENETIC ALGORITHM START ===")

    for gen in range(n_gen):

        print(f"\nGeneration {gen+1}")

        # Evaluate individuals
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        pop = offspring

        # Print best in generation
        best = tools.selBest(pop, 1)[0]

        if not best.fitness.valid:
            best.fitness.values = toolbox.evaluate(best)

        print("Best so far:", best, best.fitness.values[0])

    # Final best
    best_ind = tools.selBest(pop, 1)[0]

    print("\n=== FINAL RESULT ===")
    print("Best individual:", best_ind)
    print("Best score:", best_ind.fitness.values[0])

    return best_ind.fitness.values[0]