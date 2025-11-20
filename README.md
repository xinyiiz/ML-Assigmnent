# 📊 Machine Learning Assignment 1 – Airbnb Price Prediction & HR Analytics Clustering  
**Apr 2025 Semester**  
**Author:** Tan Xin Yi  
**Jupyter Notebook:** ML_ASG1_AY2504.ipynb

This project applies end-to-end machine learning techniques, covering both **supervised learning** and **unsupervised learning**, using real-world datasets from Airbnb and HR analytics.  
It follows the CRISP-DM workflow and demonstrates skills in data cleaning, visualization, model building, model evaluation, clustering, and interpretation.

---

# 📘 Project Overview

This assignment consists of **two major tasks**:

### **Task 1 — Supervised Learning (Airbnb Dataset)**  
Build machine learning regression models to **predict Airbnb listing prices** using listing features such as room type, number of reviews, availability, neighbourhood, and more.

### **Task 2 — Unsupervised Learning (HR Dataset)**  
Perform **clustering** on HR employee data to uncover meaningful employee groups based on attributes such as service length, training scores, performance ratings, and awards.

The project demonstrates:

- Data exploration & preprocessing  
- Feature engineering  
- Regression modeling (multiple algorithms)  
- Hyperparameter tuning  
- Clustering using K-Means & Hierarchical Clustering  
- Cluster evaluation and interpretation  
- Insights & recommendations for business decision-making  

---

# 🧠 Task 1 — Airbnb Rental Price Prediction  
_A supervised machine learning regression problem_

## 📝 1. Problem Understanding  
Airbnb hosts often struggle to set an optimal price.  
This task builds regression models that **predict the daily rental price** using listing metadata.

## 🔍 2. Data Exploration  
Using statistical summaries & visualizations, I explored relationships between:

- Room types  
- Host activity  
- Review patterns  
- Geographical factors  
- Minimum nights  
- Host listing count  

Plots include:

- Histograms  
- Boxplots for price outliers  
- Correlation heatmaps  
- Scatter plots of key predictors  

## 🧹 3. Data Cleaning & Transformation  
Key preprocessing steps:

- Handling missing values  
- Removing or capping price outliers  
- Label encoding + one-hot encoding for categoricals  
- Normalization/standardization for numerical features  
- Dropping irrelevant or high-missing-value fields  

## 🤖 4. Model Building  
I built and evaluated **at least two regression algorithms**, such as:

- **Linear Regression**  
- **Random Forest Regressor**  
- (Optional) Gradient Boosting / XGBoost  

Models were compared using:

- RMSE  
- MAE  
- R² Score  

## ⚙️ 5. Model Tuning & Improvements  
Techniques applied:

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- Feature selection  
- Testing different scaling methods  
- Trying different combinations of predictors  
- Removing skew/outliers to stabilize models  

## 🏆 6. Best Model & Findings  
I compared all models and selected the **best performing model** based on RMSE and generalization.  
Clear reasons were provided for why the chosen model outperformed others.

---

# 🧩 Task 2 — HR Analytics Promotion Segmentation  
_An unsupervised clustering problem_

## 📝 1. Problem Understanding  
HR wants to identify meaningful employee segments that may influence promotion decisions.  
The dataset includes:

- Age  
- KPIs met  
- Awards  
- Training score  
- Previous performance rating  
- Length of service  

## 🔍 2. Data Exploration  
I analyzed:

- Distribution of service length  
- Performance score patterns  
- KPI and awards relationships  
- Correlations among numerical variables  
- Outliers in training score and age  

Techniques used:

- Boxplots  
- Scatter matrices  
- Histograms  
- Pairplots  

## 🧹 3. Data Cleaning & Preparation  
- Handled missing performance ratings  
- Scaled numerical data using StandardScaler  
- Encoded categorical variables where necessary  
- Removed highly skewed variables if needed  

## 🤖 4. Clustering Models  
I built **two clustering models**:

### 🔷 **K-Means Clustering**
- Used elbow method (SSE) to determine optimal k  
- Used silhouette score to confirm cluster separation  
- Visualized clusters using 2D PCA reduction

### 🟩 **Hierarchical Clustering**
- Used Ward linkage  
- Dendrogram selection for optimal number of clusters  
- Compared cluster cohesion vs K-Means  

## 📊 5. Cluster Evaluation and Comparison  
Metrics used:

- Sum of Squared Errors (SSE)  
- Silhouette Scores  
- Visual cluster separability  

I selected the recommended clustering method and justified the choice.

## 🧠 6. Cluster Interpretation  
Each cluster was analyzed based on features like:

- Performance  
- Awards  
- Training scores  
- Age  
- Years of service  

Clusters were assigned business-meaningful names such as:

- **“High Performers”**  
- **“Experienced Consistent Workers”**  
- **“New but High-Potential Employees”**  
- **“Underperforming Segment”**  

This provides HR with actionable insights for employee development & promotion planning.

---

# 📌 Summary & Reflection

### 🔹 Summary of Findings  
I consolidated insights from both datasets:

- Key features that drive Airbnb listing prices  
- Best-performing regression model and its predictive accuracy  
- Employee clusters with similar characteristics  
- Practical implications for pricing strategy & HR decision-making  

### 🔹 Recommendations for Improvement  
- Use advanced models (XGBoost, CatBoost)  
- Apply deep learning models on Airbnb text descriptions  
- Perform more feature engineering  
- Use additional HR data such as job roles, salary bands, or tenure progression  

### 🔹 Reflection  
I reflected on:

- My progress in data science & ML  
- What I learned about model tuning and clustering  
- How generative AI assisted (and where human judgement remains essential)  


---

# 📂 Project Structure

ML_Assignment_1/
│── ML_ASG1_AY2504.ipynb # Full analysis, models & report
│── listings.csv # Airbnb dataset
│── hr_data.csv # HR dataset
│── README.md # Project documentation


---

# 🛠️ Technologies Used

- Python  
- pandas, numpy  
- scikit-learn (Regression, Clustering, Preprocessing)  
- matplotlib, seaborn  
- Jupyter Notebook  
- CRISP-DM Framework  

---

# 🎓 Skills Demonstrated

- Machine learning modeling  
- Data cleaning & preprocessing  
- Feature engineering  
- Hyperparameter tuning  
- Clustering analysis  
- Visualization & interpretation  
- Business insights development  
- Technical report writing  

---

# 🎯 Conclusion

This project demonstrates my ability to apply end-to-end machine learning workflows on real-world datasets.
Through both supervised and unsupervised learning tasks, I showcased skills in data preprocessing, feature engineering, model evaluation, clustering, and insight generation.

The Airbnb price prediction task strengthened my understanding of regression modelling and the importance of data quality in predictive accuracy.
The HR analytics task allowed me to interpret employee clusters and translate technical findings into meaningful, actionable insights.

Overall, this assignment highlights my growth as a data practitioner—capable of turning raw data into structured analysis, building reliable machine learning models, and communicating results that support informed decision-making.

