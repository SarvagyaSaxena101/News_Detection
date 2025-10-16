from transformers import TFAutoModelForSequenceClassification
from pathlib import Path
p = Path(__file__).parent / "Bert"
print('Model folder exists:', p.exists())
try:
    m = TFAutoModelForSequenceClassification.from_pretrained(str(p))
    print('Loaded HF model from folder, model class:', type(m))
    # print number of parameters
    total = sum([p.numpy().size for p in m.trainable_variables]) if m.trainable_variables else 0
    print('Trainable params count (approx):', total)
except Exception as e:
    print('Failed to load HF model from folder:', repr(e))
