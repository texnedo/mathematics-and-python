import numpy as np
from statsmodels.stats.power import TTestIndPower

# Define parameters
alpha = 0.05
power = 0.8
effect_size = 0.001 / 0.01  # (mu_2 - mu_1) / sigma
ratio = 1  # Assuming equal sample sizes in both groups

# Calculate sample size
analysis = TTestIndPower()
sample_size_per_group = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=ratio)

print(f"Required sample size per group: {sample_size_per_group:.2f}")