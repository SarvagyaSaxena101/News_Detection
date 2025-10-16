# News Bias Detection — Streamlit app

This repository includes a small Streamlit app that loads a local TensorFlow BERT model and tokenizer to predict a bias label for input text.

Files of interest
- `Bert/tf_model.h5` — TensorFlow saved Keras model file
- `Tokenizer/` — tokenizer files (e.g. `vocab.txt`)
- `label_encoder.pkl` — sklearn LabelEncoder saved with joblib
- `app.py` — Streamlit app

Quick start (Windows PowerShell)

1. Activate your virtual environment (if using the provided `myvenv`):

```powershell
. .\myvenv\Scripts\Activate.ps1
```

2. Install requirements (if not already installed):

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

Notes
- The app expects `Bert/tf_model.h5`, tokenizer files under `Tokenizer/`, and `label_encoder.pkl` to exist in the repository root. Adjust paths in `app.py` if you move them.
- If your model expects additional input names, you may need to adapt the predict function in `app.py`.
