import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import entropy, chisquare


def plot_dist(target_age, target_sex, observed_age, observed_sex): 
    # Age comparison
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    target_age.plot(kind='bar', alpha=0.8, label='Target',color='red')
    observed_age.plot(kind='bar', alpha=0.8, label='Observed', color='blue')
    plt.title("Age Distribution")
    plt.ylabel("Proportion")
    plt.legend()

    # Sex comparison
    plt.subplot(1, 2, 2)
    target_sex.plot(kind='bar', alpha=0.8, label='Target', color = 'red')
    observed_sex.plot(kind='bar', alpha=0.8, label='Observed', color='blue')
    plt.title("Sex Distribution")
    plt.ylabel("Proportion")
    plt.legend()

    plt.tight_layout()
    plt.show()


def kl_div(target_age, target_sex, observed_age, observed_sex):
# KL divergence (target || observed)
    kl_age = entropy(target_age.sort_index(), observed_age.sort_index())
    kl_sex = entropy(target_sex.sort_index(), observed_sex.sort_index())


    print(f"KL divergence (age): {kl_age:.4f}")
    print(f"KL divergence (sex): {kl_sex:.4f}")

    if (kl_age > 0.1) & (kl_sex > 0.1):
        print(f"Poor match")
    else:
        print(f"Good match")




def chi_square_test(observed_counts, expected_counts, verbose=True):

    result = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    if verbose:
        if result.pvalue < 0.05:
            print(f"Significant difference (p = {result.pvalue:.4f}) — Poor match.")
        else:
            print(f"No significant difference (p = {result.pvalue:.4f}) — Good match.")
        
    # return result.pvalue



