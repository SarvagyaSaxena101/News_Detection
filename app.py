import streamlit as st
import tensorflow as tf
import numpy as np
import os
import joblib
from pathlib import Path

# HuggingFace tokenizer (fast/vocab files)
from transformers import BertTokenizerFast
from transformers import TFAutoModelForSequenceClassification, AutoConfig
from tensorflow.keras.models import load_model as keras_load_model

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "Bert"
TOKENIZER_DIR = ROOT / "Tokenizer"
LABEL_ENCODER_PATH = ROOT / "label_encoder.pkl"


@st.cache_resource
def load_model():
    """Load the TensorFlow model.

    Strategy:
    1. Try to load a transformers TFAutoModelForSequenceClassification from the folder (`Bert/`) using
       `from_pretrained`. This works if the model was saved with the Hugging Face save_pretrained.
    2. If that fails, check for a `tf_model.h5` file. If it appears to be weights-only (no config inside),
       create a model from the config and load weights.
    3. If neither works, show a helpful Streamlit error.
    """
    # Prefer transformers loader which understands HF format
    try:
        if MODEL_DIR.exists():
            # from_pretrained will load config + weights if available
            model = TFAutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
            return model
    except Exception as e:
        # Common cause: transformers <-> Keras 3 incompatibility. Provide a clearer hint.
        msg = str(e)
        if 'Your currently installed version of Keras is Keras 3' in msg or 'install the backwards-compatible tf-keras' in msg:
            st.error("Incompatible Keras/transformers detected. Please install the compatibility package `tf-keras` or use a transformers version compatible with Keras 3. Example: `pip install tf-keras`.")
        else:
            st.warning(f"Could not load Hugging Face TF model from folder: {e}")

    # Fallback: load a Keras model or weights h5
    model_path = MODEL_DIR / "tf_model.h5"
    if not model_path.exists():
        st.error(f"Model not found at {model_path}. Expected either a HuggingFace-style folder or a tf_model.h5")
        return None

    # Try to load full Keras model first (may raise ValueError if file contains only weights)
    try:
        model = keras_load_model(str(model_path), compile=False)
        return model
    except Exception as e:
        st.warning(f"Failed loading full Keras model from h5 (may be weights-only): {e}")

    # If tf_model.h5 contains weights only, try constructing model from config + load_weights
    config_file = MODEL_DIR / "config.json"
    if config_file.exists():
        try:
            config = AutoConfig.from_pretrained(str(MODEL_DIR))
            model = TFAutoModelForSequenceClassification.from_config(config)
            # Load weights into model
            try:
                model.load_weights(str(model_path))
                return model
            except Exception as w_e:
                st.error(f"Failed to load weights from {model_path}: {w_e}")
                return None
        except Exception as c_e:
            st.error(f"Failed to build model from config in {MODEL_DIR}: {c_e}")
            return None

    st.error("Unable to load model. Ensure the `Bert/` folder contains a Hugging Face TF model (config.json + tf_model.h5 saved via `save_pretrained`) or a full Keras model saved in HDF5.")
    return None


@st.cache_resource
def load_tokenizer():
    # Use BertTokenizerFast with the tokenizer folder which contains vocab.txt
    if not TOKENIZER_DIR.exists():
        st.error(f"Tokenizer directory not found: {TOKENIZER_DIR}")
        return None
    try:
        tokenizer = BertTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
        return tokenizer
    except Exception as e:
        st.error(f"Failed to load tokenizer: {e}")
        return None


@st.cache_resource
def load_label_encoder():
    if not LABEL_ENCODER_PATH.exists():
        st.error(f"Label encoder not found at {LABEL_ENCODER_PATH}")
        return None
    return joblib.load(LABEL_ENCODER_PATH)


def preprocess(texts, tokenizer, max_length=128):
    # returns model-ready inputs (input_ids, attention_mask)
    # use numpy tensors for generality, but we'll convert to TF tensors before calling TF models
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="np")
    return enc


def predict(texts, model, tokenizer, label_encoder):
    enc = preprocess(texts, tokenizer)
    # Prepare inputs: convert numpy arrays to TF tensors for TFAutoModel
    try:
        inputs = {
            'input_ids': tf.convert_to_tensor(enc['input_ids']),
            'attention_mask': tf.convert_to_tensor(enc['attention_mask'])
        }
    except Exception:
        # fallback: try raw enc
        inputs = enc

    # Call the model. HuggingFace TF models often return a TFSequenceClassifierOutput with .logits
    try:
        output = model(**inputs, training=False)
    except Exception:
        # try predict as fallback
        try:
            output = model.predict(inputs)
        except Exception:
            # last resort: pass only input_ids
            output = model.predict(inputs.get('input_ids', enc.get('input_ids')))

    # Extract logits/tensor from HF output objects
    preds = None
    # If object has .logits attribute (HF TF output), use it
    if hasattr(output, 'logits'):
        preds = output.logits
    elif isinstance(output, (list, tuple)):
        preds = output[0]
    else:
        preds = output

    # If preds is a TF tensor, convert to numpy
    try:
        if hasattr(preds, 'numpy'):
            preds = preds.numpy()
    except Exception:
        pass

    preds = np.asarray(preds)

    # If logits/probabilities
    if preds.ndim == 2:
        labels_idx = np.argmax(preds, axis=1)
        probs = np.max(tf.nn.softmax(preds, axis=1).numpy(), axis=1)
    else:
        # binary / single-dim outputs
        if preds.ndim == 1:
            labels_idx = (preds > 0.5).astype(int).ravel()
            probs = preds.ravel()
        else:
            # unexpected shape: try argmax over last axis
            labels_idx = np.argmax(preds, axis=-1)
            probs = np.max(tf.nn.softmax(preds, axis=-1).numpy(), axis=-1)

    labels = label_encoder.inverse_transform(labels_idx)
    return list(zip(labels, probs))


def main():
    st.title("News Bias Detection (BERT)")

    st.markdown("Enter a news headline or paragraph and click Predict. The app uses a local TensorFlow BERT model.")

    model = load_model()
    tokenizer = load_tokenizer()
    label_encoder = load_label_encoder()

    if model is None or tokenizer is None or label_encoder is None:
        st.stop()

    text = st.text_area("Text to classify", height=150)
    max_len = st.sidebar.slider("Max tokens", min_value=32, max_value=512, value=128, step=32)

    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Running model..."):
                try:
                    results = predict([text], model, tokenizer, label_encoder)
                    label, prob = results[0]
                    st.success(f"Predicted: {label} (confidence: {prob:.3f})")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
