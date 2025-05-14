import numpy as np

def monte_carlo_mcal(prices, R, cov_matrix, N=10000):
    n_assets = len(prices.columns)
    mean_returns = prices.pct_change().dropna().mean().values
    A = mean_returns.reshape(1, -1)
    b = np.array([R])
    H = cov_matrix.values

    def generate_random_weights():
        weights = np.random.dirichlet(np.ones(n_assets))
        weights /= np.sum(weights)  # normalize to sum to 1
        return weights

    def risk(x):
        return x.T @ H @ x

    def constraint_violation(x):
        return np.linalg.norm(A @ x - b) ** 2

    # Step 1: Generate feasible solutions
    xset = []
    for _ in range(N):
        x = generate_random_weights()
        if np.abs(np.sum(x) - 1.0) > 1e-3:
            continue
        xset.append(x)

    # Step 2: Find xfrom that minimizes constraint violation
    xfrom = min(xset, key=lambda x: constraint_violation(x))

    xfrom_risk = risk(xfrom)
    xfrom_violation = constraint_violation(xfrom)

    # Step 3: Find xto ∈ xset such that xfromᵀHxfrom > xtoᵀHxto
    xtoset = [x for x in xset if risk(x) < xfrom_risk]

    # Step 4: For each xto, calculate Mcal
    ML = []
    for xto in xtoset:
        xto_risk = risk(xto)
        xto_violation = constraint_violation(xto)
        numerator = xfrom_risk - xto_risk
        denominator = xto_violation - xfrom_violation
        if denominator != 0:
            Mcal = numerator / denominator
            if Mcal > 0:
                ML.append(Mcal)

    # Step 5: Return the maximum Mcal
    if ML:
        M_max = max(ML)
    else:
        M_max = None  # No valid Mcal found

    print("xfrom risk:", xfrom_risk)
    print("Number of xto candidates:", len(xtoset))
    print("Max Mcal:", M_max)

    return M_max