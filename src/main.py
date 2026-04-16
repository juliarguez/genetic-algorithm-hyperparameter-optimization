from model import evaluate_model
from random_search import run_random_search

if __name__ == "__main__":

    score = evaluate_model(
        n_estimators=100,
        max_depth=5,
        min_samples_split=2
    )

    print("Baseline score:", score)



print("\n=== RANDOM SEARCH ===")
best_random = run_random_search()
