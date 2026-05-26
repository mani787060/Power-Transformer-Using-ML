# Feature Engineering: Variance Stabilization via PowerTransformer
[![Machine Learning](https://img.shields.io/badge/Domain-Feature%20Engineering-blue)](https://scikit-learn.org/)
[![Preprocessing](https://img.shields.io/badge/Transformer-PowerTransformer-orange)](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
[![Dataset](https://img.shields.io/badge/Dataset-Concrete%20Compressive%20Strength-green)](./concrete_data.csv)

## 🏗️ Project Overview
Parametric machine learning models—such as Linear Regression, Ridge, and Neural Networks—rely heavily on the assumption that numerical features are normally distributed and homoscedastic (possessing constant variance). In real-world engineering datasets like the **Concrete Compressive Strength Dataset**, components like *Fly Ash*, *Superplasticizer*, or *Age* often display multi-modal distributions, heavy tails, and severe skewness. Left uncorrected, these non-Gaussian distributions violate model assumptions, distort gradient weight updates, and degrade predictive performance.

This repository demonstrates the advanced application of Scikit-Learn's **`PowerTransformer`** to systematically map highly skewed continuous variations into stabilized, Gaussian-like normal distributions. Unlike basic stateless mathematical mappings, `PowerTransformer` functions as a stateful, parametric preprocessing engine that estimates the optimal power parameter ($\lambda$) via maximum likelihood to minimize variance distortion across data subsets.

---

## 🛠️ Advanced Engineering Mechanics

### 1. Box-Cox vs. Yeo-Johnson Transform Implementations
The `PowerTransformer` supports two foundational parametric algorithms designed to reshape underlying data distributions based on numerical constraints:

* **Box-Cox Transformation:** Formulated to find an optimal power parameter $\lambda$ using maximum likelihood estimation. It can only be applied to strictly positive data ($x > 0$).
    $$x^{(\lambda)} = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \ln(x) & \text{if } \lambda = 0 \end{cases}$$
* **Yeo-Johnson Transformation:** A generalized expansion of Box-Cox that accommodates zero and negative values ($x \le 0$), making it the standard choice for multi-feature pipelines containing zero-inflated continuous values (e.g., concrete mixtures with zero *Blast Furnace Slag* or *Fly Ash*).
    $$x^{(\lambda)} = \begin{cases} \frac{(x+1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, x \ge 0 \\ \ln(x+1) & \text{if } \lambda = 0, x \ge 0 \\ \frac{-[(-x+1)^{2-\lambda} - 1]}{2-\lambda} & \text{if } \lambda \neq 2, x < 0 \\ -\ln(-x+1) & \text{if } \lambda = 2, x < 0 \end{cases}$$

### 2. The Operational Flow
The pipeline ingests raw structural inputs, determines parameter shapes, minimizes skewness using maximum likelihood optimizations, and applies standard zero-mean, unit-variance adjustments concurrently.



```text
                 ┌─────────────────────────────────────┐
                 │     Raw Non-Gaussian Features       │
                 └─────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼

   Box-Cox Transform         Yeo-Johnson Transform    Maximum Likelihood
    (x > 0 only)           (supports zero/negative)     Estimation (λ)
        │                          │                          │
        ▼                          ▼                          ▼

 Variance Stabilization    Flexible Power Transform     Optimal Lambda Search
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼

                 ┌─────────────────────────────────────┐
                 │  Zero-Mean / Unit-Variance Matrix   │
                 └─────────────────────────────────────┘
```

---

## 🔬 Implementation Steps

The source notebook `PowerTransformer in Machine Learning.ipynb` implements a modular execution structure:

1.  **Statistical Skewness Auditing:** Utilizing Pandas `.skew()` and SciPy metrics to map baseline skewness parameters across raw concrete parameters.
2.  **Parametric Configuration:** Instantiating `PowerTransformer(method='yeo-johnson', standardize=True)` to dynamically handle the dataset's zero values.
3.  **Cross-Validation Partitioning:** Structuring transformations exclusively inside train/test boundaries to protect the calculated $\lambda$ parameter from data leakage.
4.  **Distribution Visualization Analytics:** Rendering side-by-side **Before vs. After** analysis panels using Seaborn histograms and Probability Plots (Q-Q Plots) to visually verify conformity to normal curves.
5.  **Estimator Comparison Evaluation:** Training a regression algorithm on raw data vs. power-transformed features to benchmark the direct improvements in $R^2$ and Root Mean Squared Error (RMSE) performance.

---

## 📊 Transformation Archetype Metrics

| Feature Behavior | Problem Type | Optimal Transform Method | Post-Transformation Outcome |
| :--- | :--- | :--- | :--- |
| **Strictly Positive (e.g., Price, Fare)** | Severe Skewness | Box-Cox / Yeo-Johnson | Symmetrical distribution, minimized variance |
| **Zero-Inflated (e.g., Mixture Ingredients)** | Zero-value boundaries | Yeo-Johnson (Required) | Handles absolute zero coordinates without mathematical infinity errors |
| **Heavy-Tailed Distributions** | Skewed variance spread | Yeo-Johnson / Box-Cox | Compresses extreme outliers closer to the mean |

---

## 💻 Tech Stack & Requirements
* **Language Ecosystem:** Python 3.9+
* **Data Layout Engine:** Pandas, NumPy
* **Pipeline & Modeling Framework:** Scikit-Learn (`preprocessing.PowerTransformer`, `linear_model`)
* **Statistical Analysis & Diagnostics:** SciPy (`stats.probplot`), Matplotlib, Seaborn
* **Environment Runtime:** Jupyter Notebook

---

## 🚀 Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/PowerTransformer-in-Machine-Learning.git](https://github.com/your-username/PowerTransformer-in-Machine-Learning.git)
    cd PowerTransformer-in-Machine-Learning
    ```
2.  **Install Production Environment Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
    ```
3.  **Execute the Diagnostics Pipeline:**
    ```bash
    jupyter notebook
    ```
    Open `PowerTransformer in Machine Learning.ipynb` to step through the parameter searches and distribution plots.
