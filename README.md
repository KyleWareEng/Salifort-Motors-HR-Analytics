# Salifort Motors: Predictive Employee Turnover Modeling

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning solution to predict employee turnover and enable proactive HR interventions - completed as the capstone project for the [Google Advanced Data Analytics Professional Certificate](https://www.coursera.org/professional-certificates/google-advanced-data-analytics).

![Burnout Cluster Analysis](figures/burnout_cluster.png)

---

## Executive Summary

Employee turnover costs companies **50-200% of an employee's annual salary** in recruitment, training, and lost productivity. This project addresses this critical business challenge for Salifort Motors by developing a machine learning solution to identify at-risk employees before they resign.

**The final Random Forest model achieved 92.5% Recall**, enabling HR to proactively intervene with high-performing individuals *before* they leave — potentially saving hundreds of thousands in annual recruitment costs.

### Key Results

| Metric | Value |
|--------|-------|
| **Model Recall** | 92.5% (catches 92.5% of employees who will leave) |
| **Model Precision** | 98.9% (predictions are highly accurate) |
| **F1 Score** | 95.6% |
| **Top Predictor** | Satisfaction Level (32.5% importance) |

---

## Key Insights & Visualisations

### 1. The Burnout Cluster

A critical discovery: **high-performing employees working 250+ hours/month are leaving at alarming rates**, even with strong evaluation scores. This "burnout cluster" represents the company's most valuable talent silently burning out.

### 2. Feature Importance: What Drives Turnover?

The Random Forest model identified the top predictors of employee attrition:

1. **Satisfaction Level** (32.5%) - The strongest indicator of turnover risk
2. **Tenure** (17.6%) - Highlights the critical 3-5 year danger zone
3. **Number of Projects** (17.4%) - Too few or too many projects increases risk
4. **Average Monthly Hours** (15.3%) - Identifies overworked employees
5. **Last Evaluation** (13.3%) - High performers are also at risk

![Feature Importance](figures/feature_importance.png)

### 3. Model Performance

The Random Forest model correctly identifies the vast majority of at-risk employees while maintaining high precision (few false alarms).

![Confusion Matrix](figures/confusion_matrix.png)

### 4. Correlation Analysis

Understanding relationships between variables guided feature selection and revealed multicollinearity considerations.

![Correlation Heatmap](figures/correlation_heatmap.png)

### 5. Turnover by Department

Turnover patterns across departments help HR target interventions where they're needed most.

![Turnover by Department](figures/turnover_by_department.png)

---

## The PACE Strategy Framework

This project follows the **PACE** methodology taught in the Google Advanced Data Analytics Certificate:

### 1. Plan
- **Stakeholder Alignment:** Identified HR and Executive Leadership as primary stakeholders
- **Goal Setting:** Predict employee churn (binary classification) to improve retention rates
- **Ethics & Privacy:** Established guardrails ensuring predictions support employees rather than penalise them

### 2. Analyse (EDA)
- **Data Cleaning:** Removed 3,008 duplicate entries and standardised column names
- **Outlier Analysis:** Identified 824 tenure outliers representing long-tenured employees
- **Key Discovery:** Found the "Burnout Cluster" — high performers working excessive hours who are leaving

### 3. Construct (Modelling)
- **Baseline Model:** Logistic Regression achieved only 18.3% Recall (missed 82% of leavers)
- **Champion Model:** Random Forest captured non-linear patterns with 92.5% Recall
- **Metric Priority:** Focused on Recall and F1-Score to minimise false negatives

### 4. Execute (Recommendations)
Delivered actionable insights to HR leadership (see Business Recommendations below)

---

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 83.4% | 49.7% | 18.6% | 0.27 | 0.84 |
| **Random Forest (Champion)** | **98.6%** | **98.9%** | **92.5%** | **0.96** | **0.98** |

**Why Random Forest won:** The data exhibited non-linear relationships (U-shaped project risk, burnout clusters) that Logistic Regression couldn't capture. Random Forest naturally handles these complex interactions.

---

## Business Recommendations

Based on the model's findings, I recommend the following interventions:

| Priority | Recommendation | Rationale |
|:--------:|----------------|-----------|
| **High** | Cap workloads at **5 concurrent projects** | Employees with 6+ projects leave at significantly higher rates |
| **High** | Flag employees working **250+ hrs/month** | Trigger immediate manager "stay interviews" for overworked high-performers |
| **Medium** | Implement **"3-5 Year" development program** | Turnover peaks at this tenure; offer career advancement opportunities |
| **Medium** | Review **low/medium salary bands** | These groups show disproportionately high turnover |

---

## What I Learned

- Structuring a data science project using the PACE framework
- The importance of choosing the right evaluation metric (Recall vs Accuracy) based on business context
- How to translate technical findings into actionable business recommendations
- Feature engineering and handling class imbalance in classification problems
- Communicating insights effectively to non-technical stakeholders
- Leveraging AI tools to accelerate development while maintaining code quality

---

## Technical Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Logistic Regression, Random Forest) |
| **Visualisation** | Matplotlib, Seaborn |
| **Framework** | PACE (Plan, Analyse, Construct, Execute) |
| **Development** | Jupyter Notebook, VS Code |

---

## Project Structure

```
Salifort-Motors-HR-Analytics/
│
├── figures/                           # Generated visualisations
│   ├── burnout_cluster.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── turnover_by_department.png
│
├── hr_turnover_predictor.py           # Production-ready ML pipeline
├── Salifort_Motors_HR_Analytics.ipynb # Full EDA & analysis notebook
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
└── README.md
```

---

## How to Run

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KyleWareEng/Salifort-Motors-HR-Analytics.git
   cd Salifort-Motors-HR-Analytics
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis:**
   ```bash
   python hr_turnover_predictor.py
   ```

4. **View the full analysis:**
   Open `Salifort_Motors_HR_Analytics.ipynb` in Jupyter Lab or VS Code.

### Data

The dataset is from [Kaggle's HR Analytics dataset](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction). Place the CSV file in the project directory to run the pipeline.

---

## AI-Augmented Workflow

This project utilised an **AI-augmented development workflow** to enhance productivity:

- **Code Refactoring:** Transformed raw Jupyter notebook code into production-ready, modular Python classes
- **Documentation:** AI-assisted generation of comprehensive docstrings and README content
- **Debugging:** Leveraged AI for troubleshooting complex syntax and optimising visualisations
- **Best Practices:** Applied clean code principles including type hints, logging, and error handling

This approach demonstrates proficiency in leveraging modern AI tools to accelerate data science workflows while maintaining code quality.

---

## Acknowledgments

- **[Google Advanced Data Analytics Professional Certificate](https://www.coursera.org/professional-certificates/google-advanced-data-analytics)** - Capstone project for the 7-course certificate program. [Verify my certificate](https://www.coursera.org/account/accomplishments/professional-cert/4TAGPANB8CD8)
- **Kaggle** - HR Analytics dataset
- **Anthropic Claude** - AI assistance for code refactoring and documentation

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

**Kyle Ware**

- Email: kyle.ware@outlook.com
- LinkedIn: [linkedin.com/in/kyleaware](https://linkedin.com/in/kyleaware)
- GitHub: [github.com/KyleWareEng](https://github.com/KyleWareEng)
- MEng Automotive Engineering | Data Science & Quantitative Finance

Feel free to reach out if you have questions about this project!

---

*Google Advanced Data Analytics Capstone | January 2026*

