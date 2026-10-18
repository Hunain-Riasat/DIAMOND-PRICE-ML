# Assignment 3 – Diamond Price Prediction (10-Fold Cross Validation + Web App)

**Course:** Machine Learning Fundamentals (AIC354), Fall 2026
**Student:** Muhammad Hunain (FA24-BSE-083), Section B
**Instructor:** Dr. Rao Muhammad Adeel Nawab

Task 2 of the assignment: a regression problem similar to Car Price Prediction, evaluated with 5 algorithms that the car notebook did not use, 10-fold cross validation on the full dataset, and a model deployed in a web app.

## Files

| File | What it is |
|---|---|
| `diamond_price_kfold_full_dataset_10fold.ipynb` | Google Colab notebook (already run, outputs included) |
| `diamonds.csv` | Full dataset, 53,940 diamonds (source: https://github.com/mwaskom/seaborn-data) |
| `diamond_price_model.pkl` | Trained final model (whole pipeline, 0.37 MB) |
| `streamlit-app/` | The web app (`app.py`, `requirements.txt`, `runtime.txt`, model copy) |

## Results (10-fold CV, RMSE, lower is better)

| Rank | Algorithm | Mean RMSE |
|---|---|---|
| 1 | Extra Trees | $543 (model file 682 MB, too big to deploy) |
| 2 | **Hist Gradient Boosting (deployed)** | **$551** |
| 3 | Neural Network (MLP) | $682 |
| 4 | Ridge Regression | $1,136 |
| 5 | AdaBoost | $1,403 |

## Deploy the web app (free, about 5 minutes)

1. Create a **public GitHub repository** (for example `diamond-price-ml`) and upload everything in this folder, keeping the `streamlit-app` folder as it is.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub** and fill in:
   - **Repository:** `<your-username>/diamond-price-ml`
   - **Branch:** `main`
   - **Main file path:** `streamlit-app/app.py`
4. Open **Advanced settings** and choose **Python 3.13** (the version the model was saved with), then click **Deploy**.
5. Copy the public URL (`https://<something>.streamlit.app`).
6. Paste that URL in the **very first cell** of the notebook (replace `https://YOUR-APP-NAME.streamlit.app`) and save the notebook. The assignment requires the live link at the very top.

## Run the app on your own computer first (optional)

```bash
cd streamlit-app
pip install -r requirements.txt
streamlit run app.py
```

## Submit on Google Classroom

Upload: the notebook (with the live link at the top), `diamonds.csv`, and `diamond_price_model.pkl`.
Deadline: **Sunday, October 25, 2026, 23:59**.

> If you re-run the notebook in Colab, its scikit-learn version may differ from `streamlit-app/requirements.txt`. Keep the `.pkl` from this folder for the app, or update the pinned versions in `requirements.txt` to the ones printed in Step 1 of the notebook.
