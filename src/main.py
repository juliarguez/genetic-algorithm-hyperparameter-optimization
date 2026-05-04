from model import evaluate_model
from experiments import run_experiments, print_statistics, save_results, plot_results

if __name__ == "__main__":

    print("\n=== BASELINE ===")
    baseline_score = evaluate_model(
        n_estimators=100,
        max_depth=5,
        min_samples_split=2
    )

    print("Baseline score:", baseline_score)


    ga_results, random_results = run_experiments()

    print_statistics(ga_results, "GA")
    print_statistics(random_results, "Random Search")

    save_results(ga_results, random_results)

    plot_results(ga_results, random_results)