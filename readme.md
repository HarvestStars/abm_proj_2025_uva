# ABM\_PROJ\_2025\_UVA

This project implements an Agent-Based Model (ABM) simulation of the Sugarscape environment. It supports both web-based visualization and high-efficiency parallel simulations for batch analysis. The goal is to explore dynamic behaviors and perform sensitivity analysis on model parameters.

---

## 📁 Project Structure

```
src/
├── sugar_model.py                # Core Sugarscape model logic
├── sugar_agent.py                # Agent behavior logic
├── sugar-map.txt                 # Initial sugar distribution map
├── server.py                     # Web-based visualization entry (via Solara)
├── test_experiments.py           # Standard simulation script
├── test_experiments_parallel.py  # Parallelized simulation script for faster execution
├── output/                       # Stores all simulation results
├── OFAT_alpha.py, OFAT_lambda.py, ...  # One-Factor-At-a-Time (OFAT) analysis scripts
├── density_based_SA.py, density_based_SA_parallel.py  # Sensitivity analysis scripts
├── time_plot.py                  # Generates visual result plots
```

---

## 🚀 Getting Started

### 1. Set up a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run simulation

#### Option A: Web-based visualization (Solara)

```bash
solara run src/server.py
```

#### Option B: Raw simulation output (no visualization)

* Sequential (for small experiments):

  ```bash
  python src/test_experiments.py
  ```

* Parallel (recommended for large-scale runs):

  ```bash
  python src/test_experiments_parallel.py
  ```

Simulation results will be saved to the `src/output/` directory.

---

## 📊 Post-Simulation Analysis

* **OFAT Analysis**
  Scripts starting with `OFAT_` perform one-factor-at-a-time analysis based on results stored in `output/`.

* **Sensitivity Analysis (SA)**
  `density_based_SA.py` and `density_based_SA_parallel.py` conduct density-based global sensitivity analysis. The parallel version is optimized for speed.

* **Time Series Plotting**
  Use `time_plot.py` to visualize selected metrics from the simulation output.

---

## 💡 Tips

* Use the **Solara interface** when you want to interactively explore the simulation behavior.
* Use **parallel simulation scripts** when you want to collect a large batch of data for analysis.
* All outputs are stored in the `output/` folder for consistency.

---

## 📦 Dependencies

See `requirements.txt` for required Python packages. Make sure to install them in a virtual environment as described above.

---

## 🔗 License

This project is for academic use under the University of Amsterdam's Computational Science curriculum.
