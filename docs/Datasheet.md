# 📄 Datasheet: BBO Capstone Query History and Function Evaluations


## 1. Motivation

This dataset was created as part of the **BBO Capstone Project** in the Imperial College London *Professional Certificate in Machine Learning and Artificial Intelligence* programme.

The goal of this dataset is to support a **black-box optimisation task**, where eight unknown objective functions are iteratively optimised through sequential querying.

Each week:
- One input is submitted per function  
- The system returns an output  
- The result is stored and used for future optimisation  

This dataset captures the full query history and enables:
- Transparent tracking of optimisation decisions  
- Reproducibility of the optimisation process  
- Analysis of exploration vs exploitation strategies  


## 2. Composition

The dataset consists of observations for **8 black-box functions**:

| Function | Dimensions |
|----------|-----------|
| Function 1 | 2 |
| Function 2 | 2 |
| Function 3 | 3 |
| Function 4 | 4 |
| Function 5 | 4 |
| Function 6 | 5 |
| Function 7 | 6 |
| Function 8 | 8 |

Each record contains:
- Function ID  
- Week number  
- Input vector (values between 0 and 1)  
- Output value (scalar result)  

### Data Includes:
- Initial dataset (provided by course)
- Weekly inputs (Week 1 → Week 10)
- Weekly outputs (observed values)

### Known Gaps:
- Uneven sampling across search space  
- Higher density in promising regions  
- Limited exploration in high-dimensional areas  
- Bias toward boundary values after strong signals  


## 3. Collection Process

The dataset was collected over **10 optimisation rounds**.

### Process per round:

1. Load all previous data  
2. Fit a **Gaussian Process surrogate model**  
3. Generate candidate points:
   - Global random sampling  
   - Local sampling near best point  
4. Score candidates using **Upper Confidence Bound (UCB)**  
5. Submit best candidate  

### Strategy Evolution:

- **Early rounds** → Exploration-heavy  
- **Mid rounds** → Balanced exploration + exploitation  
- **Late rounds** → Function-specific strategies  

Key observation:
- Function 5 showed strong boundary behaviour → deterministic strategy used  


## 4. Preprocessing and Uses

### Preprocessing Steps:
- Inputs constrained to `[0, 1)`  
- Values formatted to **6 decimal places**  
- Out-of-bound values clipped  

### Intended Uses:
- Bayesian optimisation experiments  
- Surrogate model training  
- Strategy analysis across iterations  
- Educational demonstration  

### Inappropriate Uses:
- Real-world benchmarking  
- General statistical conclusions  
- Non-sequential ML tasks  


## 5. Distribution and Maintenance

This dataset is maintained as part of the **BBO Capstone GitHub Repository**.

### Contents:
- Weekly input/output data  
- Python optimisation code  
- Documentation (Datasheet + Model Card)  

### Maintenance Responsibilities:
- Update weekly data  
- Ensure consistency  
- Fix formatting issues  
- Document strategy changes  

### Notes:
- Dataset is for **educational purposes only**
- Repository owner is responsible for updates  


## 📌 Summary

This dataset represents a **sequential optimisation history** rather than a static dataset.

It reflects:
- Adaptive decision-making  
- Model-driven exploration  
- Real-world constraints of black-box optimisation  

---
