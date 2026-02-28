# CYBERSECURITY

## Project Summary

This project aimed to design and deploy a complete machine learning pipeline to predict the **Attack Type** (DDoS, Intrusion, Malware) from a cybersecurity dataset containing 40,000 raw logs and 25 features :contentReference[oaicite:0]{index=0}. The workflow included exploratory data analysis, feature engineering, preprocessing, model training, evaluation, and web deployment.

Exploratory analysis revealed high-cardinality categorical variables, significant missing values in log-related columns, and a perfectly balanced target distribution, suggesting a potentially synthetic or randomly generated dataset. Feature engineering was applied to extract structured predictors from timestamps, IP addresses, ports, payload text, user and device information, and hidden boolean indicators.

Three models (Logistic Regression, Decision Tree, Random Forest) were trained using a standard preprocessing pipeline (scaling and one-hot encoding). All models achieved performance close to random baseline (~33% accuracy), with tree-based models exhibiting strong overfitting :contentReference[oaicite:1]{index=1}. These results support the hypothesis that the target variable may be independent of the available features.

The project was completed with a Streamlit web application enabling CSV upload and synthetic input generation, allowing users to test model predictions interactively and demonstrating an end-to-end applied machine learning workflow.
