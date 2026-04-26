# Customer Churn Prediction Web App


A machine learning web application that predicts whether a telecom customer is likely to cancel their subscription (churn), with real-time predictions, feature importance analysis, and actionable business recommendations.

 **[Live Demo](https://churn-predictor-9vgtzwgkk7ybgniab8ekjz.streamlit.app/)** &nbsp;|&nbsp; 

---

##  Problem Statement

Customer churn is one of the biggest challenges for telecom companies. Losing a customer costs 5–25x more than retaining one. This app helps businesses identify at-risk customers **before** they leave, enabling timely intervention.

---

##  Features

- **Real-time churn prediction** — adjust customer parameters and get instant results
- **Probability score** — not just yes/no, but a confidence percentage
- **Feature importance chart** — see which factors drive churn the most
- **Business recommendations** — actionable next steps based on risk level
- **Clean 3-column UI** — Account Info, Billing Info, and Demographics

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| ML Model | Random Forest Classifier (Scikit-learn) |
| Data Processing | Pandas, NumPy |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version Control | Git & GitHub |

---

##  Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 99% |
| Precision (Churn) | 100% |
| Recall (Churn) | 97% |
| F1-Score | 98% |

**Model:** Random Forest with 100 estimators, trained on 1000 customer records with 9 features.

---

##  Features Used for Prediction

1. **Tenure** — how long the customer has been with the company
2. **Monthly Charges** — current monthly bill amount
3. **Total Charges** — cumulative billing amount
4. **Number of Products** — services subscribed to
5. **Internet Service** — whether customer has internet
6. **Phone Service** — whether customer has phone service
7. **Senior Citizen** — demographic indicator
8. **Contract Type** — month-to-month, one year, or two year
9. **Payment Method** — electronic check, mailed check, bank transfer, credit card

---

## Project Structure

```
churn-predictor/
│
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── model/
│   ├── churn_model.pkl    # Trained Random Forest model
│   ├── le_contract.pkl    # Label encoder for contract type
│   ├── le_payment.pkl     # Label encoder for payment method
│   └── features.pkl       # Feature names list
└── README.md
```

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/rashiii201/churn-predictor.git
cd churn-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python train_model.py
```

**4. Launch the app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

##  Key Learnings

- End-to-end ML pipeline from data → model → deployment
- Saving and loading ML models with `pickle`
- Building interactive UIs with Streamlit widgets
- Deploying Python apps to the cloud
- Feature importance analysis with Random Forest

---

## Future Improvements

- [ ] Add SHAP explainability for individual predictions
- [ ] Upload your own CSV dataset for batch predictions
- [ ] Add more models (XGBoost, Logistic Regression) with comparison
- [ ] Connect to a real telecom dataset (IBM Telco)
- [ ] Add user authentication for business use

---

## About

Built by **Rashi Sharma** — B.Tech Computer Science student at HPU UIT Shimla (2023–2027), CGPA 8.73, passionate about Machine Learning and building real-world AI applications.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/rashi-sharma-99498a322)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/rashiii201)

---

*If this project helped you, consider giving it a ⭐ it means a lot!*

