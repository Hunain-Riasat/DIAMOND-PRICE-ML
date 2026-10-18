# Deployment Steps (Streamlit Community Cloud)

1. Create a **public** GitHub repository (e.g. `diamond-price-ml`) and upload everything in this folder, keeping `streamlit-app/` intact.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**.
   - Repository: `<your-username>/diamond-price-ml`
   - Branch: `main`
   - Main file path: `streamlit-app/app.py`
4. In **Advanced settings** choose **Python 3.13**, then click **Deploy**.
5. Open the app URL and click **Predict Price** with the default values. Expected result: **$6,195**.
6. Paste the URL at the top of the notebook (live link: https://diamond-price-hunain.streamlit.app/).
7. Upload the notebook, `diamonds.csv` and `diamond_price_model.pkl` to Google Classroom (deadline: Sunday, October 25, 2026, 23:59).
