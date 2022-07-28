# Everything becomes easy and intuitive from here. 
# Also, Tez keeps your code clean and readable!
# Let's import a few things.

import glob
import os

import albumentations
import timm
import torch
import torch.nn as nn
from sklearn import metrics, preprocessing, model_selection

from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from tez.datasets import ImageDataset


import pandas as pd




IMAGE_PATH = "/root/dataset/img_resized_1M/cities_instagram"
MODEL_PATH = "/root/dataset/"
INPUT_PATH = "/root/dataset/"

#INPUT_PATH = "../input/instacities1m/"
#IMAGE_PATH = "../input/instacities1m/InstaCities1M/img_resized_1M/cities_instagram"
#MODEL_PATH = "../working/"
MODEL_NAME = "vit_base_patch16_224"
#MODEL_NAME = os.path.basename(__file__)[:-3]
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 20
IMAGE_SIZE = 300
IMAGE_SIZE_MODEL=224
# Let's define a model now
# We inherit from tez.Model instead of nn.Module
# we have monitor_metrics if we want to monitor any metrics
# except the loss
# and we return 3 values in forward function.

class InstaModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model(MODEL_NAME, pretrained=True, num_classes= num_classes)
        
    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": torch.tensor(accuracy, device=device)}
    
    def optimizer_scheduler(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            patience=2,
            verbose=True,
            mode="max",
            threshold=1e-4,
        )
        return opt, sch
  
    def forward(self, image, targets=None):

        outputs = self.model(image)
        
        if targets is not None:
            loss = nn.CrossEntropyLoss()(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, None, None


dfx = pd.read_csv(INPUT_PATH + "train.csv")
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

dataset_aug = albumentations.Compose(
    [
    albumentations.Resize(IMAGE_SIZE_MODEL, IMAGE_SIZE_MODEL)
    ]
)


train_dataset = ImageDataset(
    image_paths=train_image_paths,
    targets=train_targets,
    augmentations=dataset_aug,
    backend="cv2"


)

valid_dataset = ImageDataset(
    image_paths=valid_image_paths,
    targets=valid_targets,
    augmentations=dataset_aug,
    backend="cv2"
)

model = InstaModel(num_classes=dfx.category.nunique())
es = EarlyStopping(
    monitor="valid_loss",
    model_path=os.path.join(MODEL_PATH, MODEL_NAME + ".bin"),
    patience=3,
    mode="min",
)

model = Tez(model)
config = TezConfig(
    training_batch_size=TRAIN_BATCH_SIZE,
    validation_batch_size=VALID_BATCH_SIZE,
    epochs=EPOCHS,
    step_scheduler_after="epoch",
    step_scheduler_metric="valid_loss",
    num_jobs=0
)


model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    device="cuda",
    config=config,
    callbacks=[es],
)
model.save(os.path.join(MODEL_PATH, MODEL_NAME + "_image.bin"))