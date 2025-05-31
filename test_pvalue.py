import numpy as np
from scipy import stats

# Generating synthetic data for demonstration
np.random.seed(42)
# Sample data for Group A and Group B (e.g., daily CTRs)
ctr_A = np.random.normal(loc=0.061, scale=0.01, size=900)  # Group A
ctr_B = np.random.normal(loc=0.06, scale=0.01, size=900)  # Group B

# Perform independent two-sample t-test
t_statistic, p_value = stats.ttest_ind(ctr_A, ctr_B, equal_var=False)

print(f"T-Test: t_statistic = {t_statistic:.4f}, p_value = {p_value:.4f}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in CTR.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in CTR.")