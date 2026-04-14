from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def evaluate_model(n_estimators=100, max_depth=5, min_samples_split=2):

    data = load_wine()
    X = data.data
    y = data.target

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    scores = cross_val_score(model, X, y, cv=5)

    return scores.mean()

if __name__ == "__main__":

    score = evaluate_model(
        n_estimators=100,
        max_depth=5,
        min_samples_split=2
    )

    print("Baseline accuracy:", score)