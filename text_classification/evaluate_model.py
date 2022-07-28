import pandas as pd
import numpy as np
import tez
import torch
import torch.nn as nn
import transformers
from sklearn import metrics, model_selection, preprocessing
from transformers import AdamW, get_linear_schedule_with_warmup


class BERTDataset:
    def __init__(self, text, target):
        self.text = text
        self.target = target
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.max_len = 64

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.long),
        }


class BERTBaseUncased(tez.Model):
    def __init__(self, num_train_steps, num_classes):
        super().__init__()
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, num_classes)

        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=3e-5)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.CrossEntropyLoss()(outputs, targets)

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def forward(self, ids, mask, token_type_ids, targets=None):
        _, o_2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        b_o = self.bert_drop(o_2)
        output = self.out(b_o)
        loss = self.loss(output, targets)
        acc = self.monitor_metrics(output, targets)
        return output, loss, acc
    
    def process_output(self, output):
        #output = torch.argmax(output, dim=1).cpu().detach().numpy()
        return output

def evaluate(test_dataset, model_file):
    dfx = pd.read_csv(test_dataset)
    #dfx = pd.read_csv(test_dataset,nrows=200)

    #dfx = dfx.dropna().reset_index(drop=True)
    lbl_enc = preprocessing.LabelEncoder()
    dfx.category = lbl_enc.fit_transform(dfx.category.values)

    test_dataset = BERTDataset(
        text=dfx.text.values, target=dfx.category.values
    )

    model = BERTBaseUncased(
        num_train_steps=0, num_classes=dfx.category.nunique()
    )

    #model.load("/root/dataset/model_only_text.bin", device="cpu")
    model.load("/root/dataset/"+model_file)

    preds = model.predict(test_dataset, batch_size=32, n_jobs=-1)
    #print(model.monitor_metrics(preds, dfx.category.values))
    final_preds = None
    for p in preds:
        p = p.cpu().detach().numpy()
        if final_preds is None:
            final_preds = p
        else:
            final_preds = np.vstack((final_preds, p))


    #yhatt = np.argmax(preds, axis=1)
    #yhat=list(preds)
    #print(type(predictions))
    #print(len(predictions))
    #
    #print (predictions)
    #print(type(dfx.category.values))
    #print(len(dfx.category.values))
    #print(dfx.category.values)
    np.savetxt("./predictions_text.csv", 
           final_preds,
           delimiter =", ", 
           fmt ='% s')

    #print("Precision "+ model_file+ ": " + str(metrics.accuracy_score(dfx.category.values, predictions)))



if __name__ == "__main__":
    
    #evaluate(test_dataset="/root/dataset/test.csv", model_file="model_only_text.bin")
    evaluate(test_dataset="/root/dataset/test_categorical.csv", model_file="model_only_text_categorical.bin")
    #evaluate(test_dataset="/root/dataset/test_removal.csv", model_file="model_only_text_removal.bin")
    
    #evaluate(test_dataset="/root/dataset/test.csv", model_file="model_only_text.bin")   
  
