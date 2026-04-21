# 📄 Model Card: BBO Capstone Bayesian Optimisation Strategy


## 1. Overview

**Model Name:** BBO Capstone Bayesian Optimisation Strategy  
**Type:** Sequential black-box optimisation framework  
**Version:** v1.0 (Rounds 1–10)  

This model represents a **Bayesian optimisation workflow** developed to optimise eight unknown objective functions in a constrained, black-box environment.

The approach combines:
- Gaussian Process (GP) surrogate modelling  
- Upper Confidence Bound (UCB) acquisition function  
- Hybrid candidate generation (global + local sampling)  

Unlike a static model, this is an **iterative decision-making system** that evolves after each round based on newly observed data.


## 2. Intended Use

### Suitable For:
- Black-box optimisation problems  
- Low-data optimisation scenarios  
- Sequential decision-making under uncertainty  
- Educational demonstrations of Bayesian optimisation  

### Not Suitable For:
- High-dimensional problems without adaptation  
- Highly discontinuous or adversarial functions  
- Real-time production systems without validation  
- Tasks requiring guaranteed global optimality  


## 3. Model Details and Strategy Evolution

### 🔹 Early Rounds (Weeks 1–3)
- Strong focus on **global exploration**  
- Limited reliance on surrogate model  
- Uniform candidate generation  

### 🔹 Middle Rounds (Weeks 4–7)
- Introduction of **Gaussian Process modelling**  
- Use of **UCB acquisition function**  
- Mix of:
  - Global random sampling  
  - Local sampling near best points  

### 🔹 Late Rounds (Weeks 8–10)
- Shift to **function-specific strategies**  

#### Key Observations:
- **Function 5:** Strong boundary optimum → deterministic strategy  
- **Function 7 & 8:** Consistent improvement → local exploitation  
- **Function 2:** Moderate improvement → balanced refinement  
- **Functions 1, 3, 4, 6:** Weak/unstable → exploration-focused  

### Strategy Components:
- Adaptive kappa tuning (exploration vs exploitation)  
- Local vs global candidate balancing  
- Boundary-aware sampling  


## 4. Performance

### Evaluation Metrics:
- Best observed output per function  
- Improvement across rounds  
- Stability of discovered regions  

### Summary of Results:

| Function | Behaviour |
|----------|----------|
| Function 5 | Strongest performance, clear boundary optimum |
| Function 8 | High and improving performance |
| Function 7 | Strong local optimisation gains |
| Function 2 | Moderate but consistent improvement |
| Functions 1,3,4,6 | Weak or unstable performance |

### Key Insight:
Performance is measured based on **observed maxima**, since true optima are unknown.


## 5. Assumptions and Limitations

### Key Assumptions:
- Functions are smooth enough for GP modelling  
- Nearby inputs produce related outputs  
- Best regions are worth refining  
- Exploration uncertainty reflects true unknown regions  

### Limitations:
- Candidate search is **approximate (sampling-based)**  
- No analytical optimisation of acquisition function  
- High-dimensional search remains challenging  
- Strong dependence on early sampling decisions  
- Potential over-exploitation of local optima  

### Failure Modes:
- Missing global optimum due to sampling bias  
- Overconfidence in surrogate model  
- Reduced effectiveness on irregular functions  


## 6. Ethical Considerations

This project is educational, but transparency remains critical.

### Why Transparency Matters:
- Enables reproducibility  
- Clarifies decision-making  
- Supports debugging and improvement  
- Makes optimisation behaviour interpretable  

Documenting:
- strategy changes  
- assumptions  
- limitations  

ensures that others can understand and evaluate the approach.

### Real-World Relevance:
In real-world ML systems, lack of transparency can lead to:
- poor reproducibility  
- hidden biases  
- incorrect conclusions  

This model card helps mitigate those risks.


## 📌 Summary

This optimisation approach is:

- Adaptive  
- Data-driven  
- Function-specific  
- Iterative  

It evolves from exploration to exploitation while maintaining flexibility to detect emergent behaviours in a black-box environment.

---
