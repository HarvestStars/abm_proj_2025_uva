## ABM Agent Decision Workflow (Risk Preference + Logit Model)

### 1. Compute the Expected Utility \( \mathbb{E}[U] \) of Each Strategy

- Use the risk utility function \( U(c) \) to evaluate the expected utility of each strategy based on its payoff distribution.
- This reflects the agent’s risk attitude (risk-averse, risk-neutral, or risk-seeking).

---

### 2. Use Expected Utility as the Deterministic Term \( V \) in the Logit Model

- Let \( V = \mathbb{E}[U] \), representing the "baseline value" of the strategy.

---

### 3. Introduce a Random Noise Term \( \varepsilon \)

- Captures uncertainty in decision-making, incomplete information, behavioral heterogeneity, etc.
- Controlled by the inverse temperature parameter \( \lambda \) in the Logit model:
  - \( \lambda \to \infty \): purely deterministic choice.
  - \( \lambda \to 0 \): fully random choice.

---

### 4. Compute Strategy Selection Probability \( P_i \)

- Use the softmax function to compute the probability of selecting strategy \( i \):

\[
P_i = \frac{e^{\lambda V_i}}{\sum_j e^{\lambda V_j}}
\]

---

### 5. Perform Strategy Selection via Monte Carlo Sampling

- Sample the agent’s actual strategy based on the computed probability distribution.

---

### 6. Iterate in Multi-Agent Environment

- Agents adapt strategies or utility parameters based on feedback from the environment,
  forming an evolving and dynamic agent-based system.
