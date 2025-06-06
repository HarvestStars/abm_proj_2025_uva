import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 500)

lambdas = [0.1, 0.5, 1, 3, 10]
for lam in lambdas:
    y = 1 / (1 + np.exp(-lam * x))
    plt.plot(x, y, label=f'λ={lam}')

plt.title("Logit Choice Probability (Sigmoid) vs Utility Difference")
plt.xlabel("Utility Difference (V_i - V_j)")
plt.ylabel("Choice Probability P_i")
plt.legend()
plt.grid(True)
plt.show()

def gumbel_pdf(x, mu=0, beta=1):
    z = (x - mu) / beta
    return (1 / beta) * np.exp(-(z + np.exp(-z)))

# U=V+epsilon 
# When epsilon concentrates around 0, the Gumbel distribution noise term becomes insignificant, Which leads to a deterministic choice.
# Typically, Decision is made based on the model utility V_max, while the noise term will affect the real utility U, making it V_max + epsilon.
# So, if we can remove the noise term, we can make a deterministic choice based on the maximum utility V.
x = np.linspace(-5, 10, 500) # noise term range for Gumbel distribution, which is epsilon

betas = [0.2, 0.5, 1, 2] # beta is 1/lambda
for beta in betas:
    y = gumbel_pdf(x, beta=beta)
    plt.plot(x, y, label=f'β={beta}')

plt.title("Gumbel Distribution PDF (Extreme Value Type I)")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
