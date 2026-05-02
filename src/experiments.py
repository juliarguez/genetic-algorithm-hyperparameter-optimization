from ga import run_genetic_algorithm
from random_search import run_random_search
def run_experiments(n_runs=5):

    ga_results = []
    random_results = []

    print("\n=== RUNNING GA EXPERIMENTS ===")
    for i in range(n_runs):
        print(f"\nRun {i+1}")
        score = run_genetic_algorithm()
        ga_results.append(score)

    print("\n=== RUNNING RANDOM SEARCH EXPERIMENTS ===")
    for i in range(n_runs):
        print(f"\nRun {i+1}")
        score = run_random_search()
        random_results.append(score)

    return ga_results, random_results