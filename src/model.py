import torch
import torch.nn.functional as F
from transformers import XLNetForSequenceClassification, XLNetTokenizer


class PatentClaimClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = XLNetTokenizer.from_pretrained(
            "xlnet-base-cased", do_lower_case=True
        )
        self.model = XLNetForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, max_len=256):
        """Predict the class of a given patent claim text."""
        inputs = self.tokenizer(
            text + " [SEP] [CLS]",
            return_tensors="pt",
            max_length=max_len,
            padding="max_length",
            truncation=True,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids, token_type_ids=None, attention_mask=attention_mask
            )
            logits = output.logits

        probabilities = F.softmax(logits, dim=1).cpu().numpy().flatten()
        prediction = torch.argmax(logits, dim=1).item()
        certainty = float(probabilities[prediction])
        return prediction, certainty
