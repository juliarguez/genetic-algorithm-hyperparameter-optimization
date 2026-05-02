from ga import run_genetic_algorithm
from random_search import run_random_search
import pandas as pd
import matplotlib.pyplot as plt


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


def print_statistics(results, name):

    print(f"\n=== {name} STATISTICS ===")
    print("Best:", max(results))
    print("Worst:", min(results))
    print("Average:", sum(results)/len(results))


def save_results(ga_results, random_results):

    df = pd.DataFrame({
        "GA": ga_results,
        "RandomSearch": random_results
    })

    df.to_csv("results/results.csv", index=False)


def plot_results(ga_results, random_results):

    plt.plot(ga_results, label="GA")
    plt.plot(random_results, label="Random Search")

    plt.xlabel("Run")
    plt.ylabel("Accuracy")
    plt.title("GA vs Random Search")

    plt.legend()
    plt.savefig("results/plots/comparison.png")
    plt.show()