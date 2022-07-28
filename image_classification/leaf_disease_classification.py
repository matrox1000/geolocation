# For dataset, go to: https://www.kaggle.com/c/cassava-leaf-disease-classification
# For train_folds, go to: https://www.kaggle.com/abhishek/cassava-train-folds/
import argparse
import os

#import albumentations
import pandas as pd
import tez
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn import metrics, model_selection, preprocessing
from tez.callbacks import EarlyStopping
from tez.datasets import ImageDataset
from torch.nn import functional as F

#INPUT_PATH = "../input/"
IMAGE_PATH = "/root/dataset/img_resized_1M/cities_instagram"
MODEL_PATH = "/root/dataset/"
MODEL_NAME = "efficientnet-b3"
#MODEL_NAME = os.path.basename(__file__)[:-3]
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 20
IMAGE_SIZE = 300


class LeafModel(tez.Model):
    def __init__(self, num_classes):
        super().__init__()

        self.effnet = EfficientNet.from_pretrained("efficientnet-b3")
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(1536, num_classes)
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=3e-4)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return sch

    def forward(self, image, targets=None):
        batch_size, _, _, _ = image.shape

        x = self.effnet.extract_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        outputs = self.out(self.dropout(x))

        if targets is not None:
            loss = nn.CrossEntropyLoss()(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, None, None


if __name__ == "__main__":

    dfx = pd.read_csv("/root/dataset/train.csv")
    dfx = dfx.dropna().reset_index(drop=True)
    
    dfx["path"] = dfx["category"].astype(str) + "/" + dfx["id"].astype(str) + ".jpg"
    
    lbl_enc = preprocessing.LabelEncoder()
    dfx.category = lbl_enc.fit_transform(dfx.category.values)



    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.category.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)



    train_image_paths = [os.path.join(IMAGE_PATH, x ) for x in df_train.path.values]
    valid_image_paths = [os.path.join(IMAGE_PATH, x ) for x in df_valid.path.values]
    train_targets = df_train.category.values
    valid_targets = df_valid.category.values

    train_dataset = ImageDataset(
        image_paths=train_image_paths,
        targets=train_targets,
        channel_first=False
    )

    valid_dataset = ImageDataset(
        image_paths=valid_image_paths,
        targets=valid_targets
    )

    model = LeafModel(num_classes=dfx.category.nunique())
    es = EarlyStopping(
        monitor="valid_loss",
        model_path=os.path.join(MODEL_PATH, MODEL_NAME + ".bin"),
        patience=3,
        mode="min",
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_bs=TRAIN_BATCH_SIZE,
        valid_bs=VALID_BATCH_SIZE,
        device="cuda",
        epochs=EPOCHS,
        callbacks=[es],
        fp16=True,
        n_jobs=0
    )
    model.save(os.path.join(MODEL_PATH, MODEL_NAME + "_image.bin"))
