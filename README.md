# Chicken Disease Detection

AI-assisted poultry disease image classification using EfficientNetB3 and a Streamlit interface. The application code supports image upload/camera workflows, disease information, PDF reporting, Telegram notifications, and an optional Gemini-powered poultry assistant.

## Project files

- `app.py` – main Streamlit application
- `test.py` – alternate/test application version
- `predict.py` – standalone prediction interface
- `EDA.py` – exploratory data analysis utilities
- `location.py`, `get_location.html` – location helper components
- `Dataset.ipynb`, `Modeling.ipynb` – data exploration and model development notebooks

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Place the trained `EfficientNetB3_best.pth` file in the `models/` folder, or set `MODEL_DATA_DIR` to its folder. The model file is not included in this repository.
4. Configure credentials as environment variables (see `.env.example`). Do not commit `.env` or Streamlit secrets.
5. Run the main app: `streamlit run app.py`.

## Configuration

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to enable Telegram alerts. Set `GEMINI_API_KEY` to enable the optional Gemini assistant. Set `ADMIN_USERNAME` and `ADMIN_PASSWORD` to configure admin login. Keep these values in your hosting provider's secret/environment-variable settings.

For Streamlit Community Cloud, add the corresponding values under **App settings → Secrets** or configure supported environment variables in your deployment environment.

## Important security notes

- Never commit API keys, bot tokens, passwords, `.env` files, user databases, or private reports.
- The uploaded local `users.db` and clinical report PDFs are intentionally not included because they may contain personal information.
- User authentication in the original application stores passwords directly in SQLite. Before public deployment, replace this with salted password hashing and review database access controls.
- The model weights, processed dataset, and image assets were not present as a complete deployable bundle in the uploaded files; add only files you have permission to publish.

## Disclaimer

Predictions are AI-generated screening assistance and should not replace examination and diagnosis by a qualified poultry veterinarian.
