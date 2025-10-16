# NewsLens — News Bias Detection 📰🔍


**NewsLens is a tool for analyzing political and editorial bias in news articles. It uses a fine-tuned BERT model to classify text as biased, non-biased, or inconclusive.**

This repository contains everything you need to run the Streamlit demo, as well as the notebook used for training the model.

---

## 🚀 Live Demo

https://newsdetection-4xlnxeqmqfbxqpgko6qrzy.streamlit.app/

May take some time to load the model and install all dependencies , maybe not !!!

---

## ✨ Features

-   **Bias Detection:** Classifies news text into three categories: Biased, Non-biased, or Inconclusive.
-   **Probability Score:** Provides the probability for each category.
-   **Simple Web UI:** An easy-to-use interface built with Streamlit.
-   **BERT-based Model:** Powered by a fine-tuned `bert-base-uncased` model.

---

## 🤔 How It Works

The project follows a standard machine learning workflow:

1.  **Data Loading & Preprocessing:** The `labeled_dataset.xlsx` is loaded, and the text data is cleaned and prepared for the model.
2.  **Tokenization:** The text is tokenized using the `BertTokenizer`.
3.  **Model Training:** A `TFBertForSequenceClassification` model is fine-tuned on the labeled dataset. The training process is detailed in the `training.ipynb` notebook.
4.  **Inference:** The trained model and tokenizer are loaded into the Streamlit app for real-time predictions.

Here's a visual representation of the workflow:

```
[News Article] -> [Tokenizer] -> [BERT Model] -> [Bias Prediction]
```

![Smiley GIF](Bert/smiley.gif)

---

## 🛠️ Technologies Used

-   **TensorFlow & Keras:** For building and training the deep learning model.
-   **Hugging Face Transformers:** For the BERT model and tokenizer.
-   **Streamlit:** For creating the interactive web application.
-   **Pandas & NumPy:** For data manipulation.
-   **Scikit-learn:** For metrics and the `LabelEncoder`.
-   **Jupyter Notebook:** For model training and experimentation.

---

## ⚙️ Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-link>
    cd <your-repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♀️ Usage

Once you have installed the dependencies, you can run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

This will open the app in your web browser. You can then paste any news text into the text area and click "Analyze" to see the bias prediction.

---

## 🧠 Model Training

The model was trained using the `training.ipynb` notebook. Here's a summary of the process:

-   The `labeled_dataset.xlsx` file is used as the training data.
-   The `Label_bias` column is used as the target variable.
-   The data is split into training and testing sets.
-   A `bert-base-uncased` model is fine-tuned for sequence classification with three labels.
-   The trained model is saved in the `Bert/` directory, the tokenizer in the `Tokenizer/` directory, and the label encoder as `label_encoder.pkl`.

---

## 📂 Project Structure

```
.
├── Bert/
│   ├── config.json
│   └── smiley.gif
├── Tokenizer/
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── .gitignore
├── annotations.xlsx
├── annotators.csv
├── label_encoder.pkl
├── labeled_dataset.xlsx
├── README.md
├── requirements.txt
├── streamlit_app.py
└── training.ipynb
```

---

## 🤝 Contributing

Contributions are welcome! If you have any ideas for improvements, feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## 🙏 Acknowledgments

-   The dataset used for training is from an unlabeled source.
-   The project is inspired by the need for a more transparent and unbiased news consumption.
