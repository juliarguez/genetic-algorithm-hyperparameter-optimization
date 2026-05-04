# 🧬 Genetic Algorithm for Hyperparameter Optimization

A machine learning optimization project using Genetic Algorithms (DEAP) to tune hyperparameters of a Random Forest classifier and compare its performance against Random Search.

---

## Objective

The goal of this project is to explore whether a Genetic Algorithm can outperform Random Search in hyperparameter optimization for supervised learning models.

---

## Approach

Two optimization strategies were implemented and compared:

### Genetic Algorithm (DEAP)
- Population-based search
- Selection, crossover, and mutation operators
- Fitness based on cross-validation accuracy

### Random Search
- Baseline method for comparison
- Random sampling of hyperparameter space

---

## Model & Dataset

- Model: RandomForestClassifier (Scikit-learn)
- Dataset: Wine classification dataset
- Evaluation: 5-fold cross-validation accuracy

---

## Hyperparameters Optimized

- `n_estimators`
- `max_depth`
- `min_samples_split`

---

## Results

The project runs multiple experiments and compares:

- Genetic Algorithm performance across runs
- Random Search baseline performance
- Stability vs peak performance trade-offs

Visualization plots are generated automatically.

---

## Tech Stack

- Python
- Scikit-learn
- DEAP (Genetic Algorithms)
- Pandas
- Matplotlib

---

## How to Run

```bash
pip install -r requirements.txt
python src/main.py