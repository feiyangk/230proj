# PAN-NAN Fusion Model: Results Analysis Outline

## 1. Results Analysis

### 1.1 Overall Forecasting Performance

- **Validation Performance Metrics**
  - MAE, RMSE, directional accuracy on validation set
  - Comparison with naïve baseline (random-walk)
  - Percentage improvements over baseline
  - Interpretation: PAN-NAN fusion's ability to capture multi-source signals (price dynamics via PAN branch, sentiment via NAN branch)

- **Key Strengths Highlighted**
  - Dual-branch architecture: PAN processes price/synthetic features, NAN processes GDELT sentiment
  - Fusion mechanism combines complementary information sources
  - High directional accuracy indicates effective signal integration
  - Multi-horizon forecasting capability (7, 14, 28 days)

- **Interpretation Framework**
  - PAN branch captures price momentum, technical indicators, cross-asset relationships
  - NAN branch captures sentiment-driven inflation expectations
  - Fusion layer learns optimal weighting between price and sentiment signals
  - Directional accuracy reflects model's ability to identify regime changes

### 1.2 Benchmark Comparison (Validation Set)

**Comparison Table:**
- PAN-NAN-With-Fincast
- PAN-NAN-No-Fincast
- TFT-Final
- TFT-No-Fincast
- LSTM Baseline
- Technical baselines (MA 5/10/21, EMA)
- Naïve baseline

**Metrics to Compare:**
- MAE, RMSE, Directional Accuracy
- Relative performance vs. naïve baseline
- Relative performance vs. TFT variants
- Relative performance vs. LSTM

**Interpretation Points:**
- PAN-NAN fusion vs. single-branch models (TFT, LSTM)
- Impact of FinCast backbone on PAN branch performance
- Sentiment integration (NAN branch) vs. models without sentiment
- Multi-source fusion vs. single-source models

### 1.3 Test Set Performance and Robustness

- **Test Set Metrics**
  - Overall test performance (MAE, RMSE, directional accuracy)
  - Comparison with validation performance (generalization gap analysis)
  - Performance during different market regimes (if applicable)

- **Robustness Analysis**
  - Performance stability across different time periods
  - Handling of structural breaks or regime changes
  - Comparison with baselines on test set
  - Error distribution analysis

- **Interpretation**
  - How well PAN-NAN fusion generalizes to unseen data
  - Whether fusion mechanism maintains effectiveness on test set
  - Comparison of PAN vs. NAN branch contributions on test set
  - FinCast impact on generalization

### 1.4 Horizon-Specific Performance

**Per-Horizon Analysis:**
- H7 MAE, RMSE, directional accuracy
- H14 MAE, RMSE, directional accuracy
- H28 MAE, RMSE, directional accuracy

**Comparison Across Horizons:**
- Error growth pattern (short-term vs. long-term)
- Directional accuracy degradation with horizon
- Comparison with baselines at each horizon
- PAN vs. NAN branch performance by horizon

**Interpretation:**
- Short-term predictability (H7) - price signals dominate
- Medium-term (H14) - balanced fusion of price and sentiment
- Long-term (H28) - sentiment may become more important
- Error accumulation patterns
- Horizon-specific strengths of PAN vs. NAN branches

### 1.6 Key Takeaways (Qualitative Analysis)

**Architectural Insights:**

1. **Dual-Branch Design Effectiveness**
   - PAN branch excels at capturing price momentum and technical patterns
   - NAN branch captures sentiment-driven inflation expectations
   - Fusion mechanism learns optimal weighting between complementary signals
   - Multi-source integration outperforms single-source models

2. **FinCast Backbone Analysis**
   - FinCast impact on PAN branch performance
   - Trade-off between pre-trained features and model complexity
   - When FinCast helps vs. when raw features suffice
   - Computational cost vs. performance gain

3. **Sentiment Integration (NAN Branch)**
   - GDELT sentiment features add predictive value
   - Sentiment more important for longer horizons
   - NAN branch's transformer encoder effectively processes temporal sentiment patterns
   - Conv1D + Transformer + BiGRU pipeline captures sentiment dynamics

4. **Fusion Mechanism Behavior**
   - Learned fusion weights reveal relative importance of PAN vs. NAN
   - Fusion adapts to different market regimes
   - Fusion layer's MLP learns non-linear combinations effectively

5. **Horizon-Specific Patterns**
   - Short-term (H7): Price signals dominate, PAN branch more important
   - Medium-term (H14): Balanced fusion of price and sentiment
   - Long-term (H28): Sentiment may become more predictive
   - Error growth patterns reflect diminishing signal strength

6. **Comparison with Baselines**
   - PAN-NAN fusion vs. TFT: Dual-branch design captures multi-modal signals better
   - PAN-NAN fusion vs. LSTM: Attention mechanisms provide better temporal modeling
   - Multi-source fusion vs. single-source: Complementary signals improve forecasting

7. **Limitations and Future Directions**
   - Model performance during structural breaks or regime changes
   - Computational complexity of dual-branch architecture
   - FinCast dependency and alternative backbone options
   - Sentiment feature quality and coverage
   - Fusion mechanism interpretability

**Qualitative Conclusions:**
- PAN-NAN fusion successfully integrates price and sentiment signals
- Dual-branch architecture provides complementary information sources
- FinCast backbone enhances price feature representation when enabled
- Sentiment integration (NAN branch) adds value, especially for longer horizons
- Fusion mechanism effectively combines multi-modal signals
- Model demonstrates strong performance relative to baselines while maintaining interpretability through branch separation

---

### 2.1 Attention Visualization Analysis
- PAN branch attention patterns (MMSA layers)
- What the model attends to in price/synthetic features
- Temporal attention patterns across lookback window
- Comparison of attention across different market regimes

