# Diamond Price Prediction — Comparing 5 ML Algorithms with 10-Fold Cross Validation

An end-to-end **Machine Learning project for diamond price prediction**, developed as part of **Machine Learning Fundamentals (AIC354), Assignment 3** at **COMSATS University Islamabad, Lahore Campus**.

The project applies, evaluates and compares five regression algorithms on a large real-world dataset using **10-Fold Cross Validation** on Google Colab, and deploys the selected model as a Streamlit web application.

### Live Demo

**[Diamond Price Predictor — Streamlit App](https://diamond-price-hunain.streamlit.app/)**

---

## Project Information

| Detail                     | Information                                         |
| -------------------------- | --------------------------------------------------- |
| **Course**                 | Machine Learning Fundamentals (AIC354)              |
| **Assignment**             | Assignment 3                                        |
| **University**             | COMSATS University Islamabad, Lahore Campus         |
| **Student**                | Hunain                                              |
| **Registration No.**       | FA24-BSE-083                                        |
| **Instructor**             | Dr. Rao Muhammad Adeel Nawab                        |
| **Problem Type**           | Regression (predict diamond price in US dollars)    |
| **Deployed Model**         | Hist Gradient Boosting Regressor                    |
| **Dataset**                | Diamonds dataset (53,940 diamonds, `seaborn-data`)  |
| **Validation**             | 10-Fold Cross Validation, metric: RMSE              |
| **Cross-Validation RMSE**  | **$551** (deployed model)                           |
| **Application**            | Streamlit                                           |

---

## Project Overview

The objective of this project is to compare multiple machine learning algorithms on the same dataset and choose the best one for deployment. The problem is similar to the instructor's Car Price Prediction System, but uses a different dataset (about 4.5 times larger) and **five algorithms that were not used in the car notebook**.

The implementation follows the workflow of the course notebooks:

```text
Import Libraries
       ↓
Load Data (upload diamonds.csv in Colab)
       ↓
Data Preprocessing (duplicates, impossible values)
       ↓
Pre-processing Pipeline (impute, scale, one-hot encode)
       ↓
10-Fold Cross Validation of 5 Algorithms
       ↓
Comparison (tables, charts, observations)
       ↓
Final Model Selection and Training
       ↓
Save Model (.pkl)
       ↓
Application Phase
       ↓
Feedback Phase
       ↓
Deployment (Streamlit)
```

---

## Features Used for Prediction

The model uses **9 input features** (6 numeric, 3 categorical).

| User Input   | Dataset Column | Description                          | Application Values            |
| ------------ | -------------- | ------------------------------------ | ----------------------------- |
| **Carat**    | `carat`        | Weight of the diamond                | 0.2 – 5.1                     |
| **Depth %**  | `depth`        | Total depth percentage               | 43 – 79                       |
| **Table %**  | `table`        | Width of the top relative to widest  | 43 – 95                       |
| **Length x** | `x`            | Length in mm                         | 3 – 11                        |
| **Width y**  | `y`            | Width in mm                          | 3 – 59                        |
| **Depth z**  | `z`            | Depth in mm                          | 1 – 32                        |
| **Cut**      | `cut`          | Quality of the cut                   | Fair, Good, Very Good, Premium, Ideal |
| **Colour**   | `color`        | Diamond colour (D best, J worst)     | J, I, H, G, F, E, D           |
| **Clarity**  | `clarity`      | Clarity grade                        | I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF |

---

## Dataset

The project uses the classic **Diamonds** dataset (53,940 rows, 10 columns), available in the `seaborn-data` repository: https://github.com/mwaskom/seaborn-data

### Dataset Processing

1. Loading the dataset with Pandas (`diamonds.csv`, uploaded in Colab).
2. Removing **146 duplicate rows** (leaves 53,794 unique diamonds), so a diamond cannot appear in both training and test folds.
3. Turning impossible zero values in `x`, `y`, `z` into missing values.
4. Filling missing numeric values with the median **inside the pipeline**, using training folds only.
5. Standard scaling for numeric features and one-hot encoding for categorical features, also inside the pipeline (no data leakage).
6. Splitting the data into 10 folds (`shuffle=True`, `random_state=42`), the same folds for every algorithm.

---

## Machine Learning Algorithms

Five algorithms were compared. None of them were used in the instructor's Car Price notebook.

| # | Algorithm                  | Short description                                                     |
| - | -------------------------- | --------------------------------------------------------------------- |
| 1 | **Ridge Regression**       | Linear model with an L2 penalty on the weights                        |
| 2 | **AdaBoost**               | Boosting of 50 small trees, each focusing on previous mistakes        |
| 3 | **Extra Trees**            | 100 trees with randomly chosen split points, averaged                 |
| 4 | **Hist Gradient Boosting** | Fast histogram-based gradient boosting                                |
| 5 | **Neural Network (MLP)**   | Two hidden layers (64 and 32 neurons)                                 |

All algorithms use default settings (no hyper-parameter tuning) with `random_state=42`.

---

## Model Performance

### 10-Fold Cross Validation Results (RMSE, lower is better)

| Rank | Algorithm                  | Mean RMSE | Std RMSE | RMSE as % of Avg Price | Time (s) |
| ---- | -------------------------- | --------: | -------: | ---------------------: | -------: |
| 1    | **Extra Trees**            | **$543**  | $20      | 13.8%                  | 88.4     |
| 2    | **Hist Gradient Boosting** | $551      | $27      | 14.0%                  | 7.5      |
| 3    | Neural Network (MLP)       | $682      | $144     | 17.3%                  | 143.9    |
| 4    | Ridge Regression           | $1,136    | $36      | 28.9%                  | 1.1      |
| 5    | AdaBoost                   | $1,403    | $26      | 35.7%                  | 31.1     |

The average diamond price in the data is **$3,933**.

### Which model was deployed, and why

**Extra Trees** has the lowest RMSE, but its saved model file is about **682 MB**, which is too large for GitHub and free cloud hosting. **Hist Gradient Boosting** is only **$8 worse** (well inside one standard deviation), trains about 10 times faster, and its saved model is only **0.37 MB**. It was therefore trained on all 53,794 diamonds and deployed.

> The reported performance depends on the selected features, preprocessing, folds and default model settings used in this academic project.

---

## Project Structure

```text
Diamond-Price-Prediction/
│
├── diamond_price_kfold_full_dataset_10fold.ipynb
├── diamonds.csv
├── diamond_price_model.pkl
│
├── streamlit-app/
│   ├── app.py
│   ├── diamond_price_model.pkl
│   ├── requirements.txt
│   └── runtime.txt
│
├── Assignment_3_Documentation.docx
├── DEPLOY_STEPS.md
└── README.md
```

---

## Streamlit Web Application

The trained model is integrated into a **Streamlit** application. Users enter the six measurements and pick the cut, colour and clarity, and the app returns the estimated price in US dollars together with the typical error of the model (about $551).

### Live Demo

**[Diamond Price Predictor — Streamlit App](https://diamond-price-hunain.streamlit.app/)**

---

## Running the Notebook on Google Colab

1. Open https://colab.research.google.com and sign in.
2. **File → Upload notebook** and choose `diamond_price_kfold_full_dataset_10fold.ipynb`.
3. **Runtime → Run all**.
4. When **Step 2.1** runs, click **Choose Files** and select `diamonds.csv`.

The notebook takes roughly 7 to 10 minutes to run, mostly for Extra Trees and the neural network.

---

## Running the Streamlit Application Locally

```bash
cd streamlit-app
pip install -r requirements.txt
streamlit run app.py
```

The application opens in the browser through the local Streamlit server.

---

## Technologies Used

### Programming Language

* **Python**

### Machine Learning

* **Scikit-learn** (Ridge, AdaBoost, Extra Trees, Hist Gradient Boosting, MLP)

### Data Processing and Charts

* **Pandas**
* **NumPy**
* **Matplotlib**
* **tqdm** (progress bar)

### Model Persistence

* **Pickle**

### Development Environment

* **Jupyter Notebook / Google Colab**

### Web Application

* **Streamlit**

### Deployment

* **Streamlit Community Cloud**

---

## Key Learning Outcomes

* Handling a large real-world dataset
* Preventing data leakage with a pre-processing pipeline
* K-Fold Cross Validation
* Applying and comparing multiple algorithms on the same folds
* Evaluating regression models with RMSE
* Choosing a model for deployment using accuracy, speed and file size
* Saving a trained model and integrating it into a web application
* Deploying a machine learning application to the cloud

---

## Disclaimer

This project is intended **strictly for educational and academic purposes** as part of the Machine Learning Fundamentals course. Predicted prices reflect patterns in the public diamonds dataset and **must not be used for real pricing or valuation decisions**.

---

## Author

**Muhammad Hunain Riasat**

BS Software Engineering
COMSATS University Islamabad, Lahore Campus

**Registration No.:** FA24-BSE-083

---

## License

This project is intended for educational purposes.
