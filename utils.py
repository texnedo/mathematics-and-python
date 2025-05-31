# -*- coding: utf-8 -*-
# +
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

FONT_SIZE_TICKS = 14
FONT_SIZE_TITLE = 20
FONT_SIZE_AXES = 16


def calculate_feature_importance(features, lr, X_test, y_test):
    if len(features) > 1:
        bunch = permutation_importance(
            lr, X_test, y_test, n_repeats=10, random_state=42
        )
        imp_means = bunch.importances_mean
        ordered_imp_means_args = np.argsort(imp_means)[::-1]

        results = {}
        for i in ordered_imp_means_args:
            name = list(X_test.columns)[i]
            imp_score = imp_means[i]
            results.update({name: [imp_score]})

        most_important = list(X_test.columns)[ordered_imp_means_args[0]]
        results_df = pd.DataFrame.from_dict(results)

        return most_important, results_df

    else:
        return features[0], None


def plot_feature_importance(df):
    # Create a plot for feature importance
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Importance Score", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Feature", fontsize=FONT_SIZE_AXES)
    ax.set_title("Feature Importance", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)

    sns.barplot(data=df, orient="h", ax=ax, color="deepskyblue")

    plt.show()