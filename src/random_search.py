import random
from model import evaluate_model


def run_random_search(n_iter=20):

    best_score = -1
    best_params = None

    for i in range(n_iter):

        # random hyperparameters
        n_estimators = random.randint(10, 200)
        max_depth = random.randint(1, 20)
        min_samples_split = random.randint(2, 10)

        score = evaluate_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )

        print(f"Iter {i+1}: score={score:.4f}")

        if score > best_score:
            best_score = score
            best_params = (n_estimators, max_depth, min_samples_split)

    print("\nBest Random Search result:")
    print("Score:", best_score)
    print("Params:", best_params)

    return best_score