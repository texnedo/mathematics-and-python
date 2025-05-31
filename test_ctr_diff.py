import numpy as np
from scipy import stats

# Data
impressions_A = 1000
clicks_A = 100
impressions_B = 1200
clicks_B = 80

# CTR
ctr_A = clicks_A / impressions_A
ctr_B = clicks_B / impressions_B

# Pooled proportion
pooled_prob = (clicks_A + clicks_B) / (impressions_A + impressions_B)

# Standard error
se = np.sqrt(pooled_prob * (1 - pooled_prob) * (1/impressions_A + 1/impressions_B))

# Z-score
z = (ctr_A - ctr_B) / se

# Two-tailed p-value
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

# Print results
print(f"CTR_A: {ctr_A:.4f}")
print(f"CTR_B: {ctr_B:.4f}")
print(f"Pooled proportion: {pooled_prob:.4f}")
print(f"Standard error: {se:.4f}")
print(f"Z-score: {z:.4f}")
print(f"P-value: {p_value:.4f}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in CTR.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in CTR.")
