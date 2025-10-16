from pathlib import Path
import app
from importlib import reload
reload(app)

model = app.load_model()
if model is None:
    print('Model failed to load, aborting test')
    raise SystemExit(1)

tokenizer = app.load_tokenizer()
if tokenizer is None:
    print('Tokenizer failed to load, aborting test')
    raise SystemExit(1)

le = app.load_label_encoder()
if le is None:
    print('Label encoder failed to load, aborting test')
    raise SystemExit(1)

print('Running a sample prediction...')
res = app.predict(['This is a test headline about politics and opinion.'], model, tokenizer, le)
print('Prediction result:', res)
