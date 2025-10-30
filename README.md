# Multi-output Event Classification with DistilBERT

## Problem statement

This project trains a multi-output text classification model on event log text to predict multiple structured fields for each event. Given an `event_text` string, the model predicts:

- eventClass (e.g., System, Application, Authentication, ...)
- eventDeviceCat (device category)
- eventOperation (e.g., system event, user login, connection, ...)
- eventOutcome (Success, Failure, Normal, Alert, ...)
- eventSeverity (numeric severity; treated as a classification task)

Predictions are produced with per-field confidence scores and saved to `output_with_confidence.json`. Logging of per-event predictions is written to `prediction_logs.txt`.

## Approach

1. Data ingestion
   - Load `train_data.csv` into a pandas DataFrame.
   - Ensure `event_text` is string type and encode categorical targets with `sklearn.preprocessing.LabelEncoder` for each output field.

2. Tokenization
   - Use `DistilBertTokenizer.from_pretrained("distilbert-base-uncased")` with padding/truncation and `max_length=128`.

3. Dataset and DataLoader
   - Implement a `LogDataset` (torch.utils.data.Dataset) that tokenizes texts and stores label vectors for the 5 target fields.
   - Use a custom `collate_fn` to stack `input_ids`, `attention_mask`, and labels into batches.

4. Model architecture
   - `MultiOutputDistilBERT` wraps `DistilBertModel` and attaches one linear classifier head per target field:
     - `eventClass`, `eventDeviceCat`, `eventOperation`, `eventOutcome`: classification heads sized to the number of unique labels per field.
     - `eventSeverity`: classifier head with 6 outputs (assumes severity labels in range 1–5; 6 output neurons used to cover index space).
   - The model returns a dict of logits for each field.

5. Training
   - Loss is computed as the sum of `cross_entropy` losses for each head.
   - Optimizer: Adam with learning rate 2e-5.
   - Training loop runs (in the notebook) for 15 epochs with `batch_size=8`.

6. Saving
   - Trained model `state_dict` saved to `Model/multioutput_distilbert.pt`.
   - Tokenizer saved via `tokenizer.save_pretrained("Model/")`.
   - Label encoders saved with `joblib.dump(..., "Model/{field}_encoder.pkl")`.

7. Validation / Inference
   - During validation the notebook computes softmax confidences per field, picks argmax as prediction, inverse-transforms label encoder indices back to string labels, and writes results with confidence to `output_with_confidence.json`.

## Model used

- HuggingFace DistilBERT: `distilbert-base-uncased` as the base model (via `transformers` package).
- Fine-tuned on the event log dataset with a small fully-connected head per target field (multi-task / multi-output classification).

Saved artifacts (after running the notebook):

- `Model/multioutput_distilbert.pt` — saved PyTorch `state_dict` of the fine-tuned model.
- `Model/` — tokenizer files (`vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`) and `Model/{field}_encoder.pkl` files for label encoders.

## Installation & run commands (Windows / cmd)

1. (Optional) Activate the provided virtual environment (the repo includes a `bert/` venv). From the project root run:

```cmd
:: activate the provided environment
c:\> cd c:\Users\karmp\OneDrive\Desktop\python\models\BERTclass
c:\Users\karmp\...\BERTclass> .\bert\Scripts\activate.bat
```

Or create and activate a fresh virtual environment:

```cmd
c:\Users\karmp\...\BERTclass> python -m venv venv
c:\Users\karmp\...\BERTclass> venv\Scripts\activate.bat
```

2. Install dependencies (project includes `requirements.txt`):

```cmd
(venv) c:\Users\karmp\...\BERTclass> pip install -r requirements.txt
```

3. Run the training notebook

- Start Jupyter and open `app.ipynb`, then run cells sequentially:

```cmd
(venv) c:\Users\karmp\...\BERTclass> jupyter notebook app.ipynb
```

- Or run interactively in Jupyter Lab / VS Code Notebook.

Notes:
- If you have a GPU and PyTorch built with CUDA, the notebook will use it automatically (`torch.device("cuda" if available else "cpu")`).
- Training hyperparameters (epochs, batch_size, lr) are in the notebook and can be tuned.

## Sample input (from `train_data.csv`)

CSV row (columns: event_id, event_text, eventClass, eventDeviceCat, eventOperation, eventOutcome, eventSeverity):

```csv
230,"An TLS 1.2 connection request was received from a remote client application, but none of the cipher suites supported ...",Application,System,app event,Failure,3
```

(Shortened event_text for readability.)

## Sample output (one entry from `output_with_confidence.json`)

This is the structure written by the notebook during validation / inference:

```json
{
  "event_id": 1,
  "eventClass": "System",
  "eventDeviceCat": "System",
  "eventOperation": "system event",
  "eventOutcome": "Normal",
  "eventSeverity": 1,
  "confidence": {
    "eventClass": 0.5271725058555603,
    "eventDeviceCat": 0.7520215511322021,
    "eventOperation": 0.499851256608963,
    "eventOutcome": 0.9298281073570251,
    "eventSeverity": 0.8450760245323181
  }
}
```

Each prediction contains the label and a confidence score for each field (float in [0,1]).

## Files produced by the notebook

- `Model/multioutput_distilbert.pt` — model weights
- `Model/` — tokenizer files and saved label encoders (`{field}_encoder.pkl`)
- `output_with_confidence.json` — JSON array of predicted events with confidences
- `prediction_logs.txt` — line-by-line logs recorded via Python `logging`

## Caveats & next steps

- Severity head in the notebook uses a Linear layer with 6 outputs (assumes severity values in a 1–5 range). Verify label encoder mapping to ensure indices align with the head size.
- The notebook uses a simple sum of cross-entropy losses across heads. Consider weighting losses if some tasks are more important or unbalanced.
- For production inference, consider creating a small Python script that loads tokenizer, model, and encoders and offers a CLI / REST wrapper.
- Consider using a proper training/validation metrics suite (accuracy per head, F1 for imbalanced classes) and saving best-checkpoint by validation metric.

## Contact / Troubleshooting

If you run into issues:
- Confirm the Python environment has matching versions of `torch` and `transformers`.
- If memory/GPU errors occur, lower `batch_size` or use gradient accumulation.
- If training is very slow, use a machine with a CUDA-capable GPU and an appropriate PyTorch install.

