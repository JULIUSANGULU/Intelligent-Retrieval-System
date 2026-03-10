import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("results/experiment_results.csv")

models = ["Boolean", "TF-IDF", "Semantic"]

precision_scores = [
    results["boolean_precision"].mean(),
    results["tfidf_precision"].mean(),
    results["semantic_precision"].mean()
]

recall_scores = [
    results["boolean_recall"].mean(),
    results["tfidf_recall"].mean(),
    results["semantic_recall"].mean()
]

# Precision Graph
plt.bar(models, precision_scores)
plt.title("Precision Comparison")
plt.ylabel("Precision")
plt.savefig("results/precision_comparison.png")
plt.clf()

# Recall Graph
plt.bar(models, recall_scores)
plt.title("Recall Comparison")
plt.ylabel("Recall")
plt.savefig("results/recall_comparison.png")