![tierra-mallorca-rgJ1J8SDEAY-unsplash](https://github.com/user-attachments/assets/15f10963-f38f-4de1-b5eb-a6d60a155678)


# House Price Prediction using ML

By

**Dudekula Abid Hussain**

Email - dabidhussain2502@gmail.com | Kaggle - https://www.kaggle.com/abiabid 

This project is based on the classic **House Prices: Advanced Regression Techniques** [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition from **Kaggle**, where the objective is to predict house sale prices using various features. The dataset contains **1460 rows** and **80 features** representing physical attributes and location-based characteristics of houses.

**Problem Statement** - Build a regression model to accurately predict the **Sale Price** of houses using the given data. Since housing prices are influenced by a combination of numeric and categorical variables, proper preprocessing and modeling are key to achieving high prediction accuracy.


## Tools & Libraries Used

* Python (Jupyter Notebook)
* pandas, numpy
* seaborn, matplotlib
* scikit-learn
* xgboost
* Pipeline, GridSearchCV, cross\_val\_score


## EDA & Preprocessing

* Statistical Summary and Null Value Handling
* **Correlation Heatmap** to detect multicollinearity
* Visual analysis showed **right-skewed distribution** for house prices

### Categorical Encoding:

* **OneHotEncoding** → For linear models (Ridge, Lasso)
* **LabelEncoding** → For tree-based models (Decision Tree, Random Forest, XGBoost)

### Target Variable Transformation:

* Applied `np.log1p(SalePrice)` to normalize right-skewed prices
* This improved model stability and reduced RMSE


## Modeling Workflow

All models were wrapped in **sklearn Pipelines** and tuned using **GridSearchCV** to avoid overfitting and automate preprocessing.

### 1. Ridge Regression

* **Best Alpha**: 50
* **Best CV RMSE**: `0.1570`

### 2. Lasso Regression

* **Best Alpha**: 0.01
* **Best CV RMSE**: `0.1689`

### 3. Decision Tree

* Best Parameters:
  `{max_depth: 20, min_samples_leaf: 4, min_samples_split: 15}`
* **Best CV RMSE**: `0.1962`

### 4. Random Forest

* Best Parameters:
  `{n_estimators: 100, max_depth: 20, max_features: 'sqrt'}`
* **Best CV RMSE**: `0.1464`

### 5. XGBoost

* Best Parameters:
  `{n_estimators: 200, max_depth: 3, colsample_bytree: 0.8, subsample: 0.8}`
* **Best CV RMSE**: `0.1366`

**XGBoost was selected as the best-performing model.**


## Insights

* **Tree-based models** (especially Random Forest and XGBoost) outperformed linear models.
* **Log transformation** of target variable significantly improved performance.
* Proper encoding of categorical variables is critical depending on the model type.



## Conclusion

This project shows how a robust machine learning pipeline can accurately estimate house prices using preprocessing, encoding, model tuning, and validation. It’s a great foundation for solving real-world regression problems and building deployable ML apps.



---

Thanks for stopping by! If you found this helpful or have suggestions, feel free to leave feedback. Happy learning and exploring new data!

