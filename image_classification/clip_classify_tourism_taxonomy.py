import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing, model_selection
import os

IMAGE_PATH = "/root/dataset/img_resized_1M/test"
MODEL_PATH = "/root/dataset/"
INPUT_PATH = "/root/dataset/"



def compute_similarity(image_features, prompt_features):
    return (
        (100.0 * image_features @ prompt_features.T)
        .softmax(dim=-1)
        .detach()
        .cpu()
        .numpy()
    )




dfx = pd.read_csv(INPUT_PATH + "test.csv")
dfx = dfx.dropna().reset_index(drop=True)
dfx["path"] = dfx["category"].astype(str) + "/" + dfx["id"].astype(str) + ".jpg"
    
lbl_enc = preprocessing.LabelEncoder()
test_targets_names = dfx.category.values
dfx.category = lbl_enc.fit_transform(dfx.category.values)



test_image_paths = [os.path.join(IMAGE_PATH, x ) for x in dfx.path.values]
test_targets = dfx.category.values

#imagespath = 'InstaNY100K/img_resized/newyork/*.jpg'
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
text_file = open(INPUT_PATH + "labelsV1_5.txt", "r")
labels = text_file.readlines()
texts = [f"A photo of  {i}" for i in labels]
prompts = clip.tokenize(labels).to(device)
with torch.no_grad():
  prompt_features = model.encode_text(prompts)
  prompt_features /= prompt_features.norm(dim=-1, keepdim=True)

final_preds = ["id","tourism_taxonomy","probability","city_id","city_name"]
for file,target in zip(test_image_paths,test_targets):
  id=os.path.basename(file).split(".")[0]
  #print(file)

  with torch.no_grad():
    image = preprocess(Image.open(file)).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
  image_features /= image_features.norm(dim=-1, keepdim=True)
  similarity = compute_similarity(image_features, prompt_features)
    
  sal = labels[np.argmax(similarity[0])]
  sal2= np.max(similarity[0])

  p=[id,sal,sal2,target,lbl_enc.classes_[target]]
  final_preds = np.vstack((final_preds, p))  
  
  #print(f"{id}: {sal} ({sal2}), city: {target}({lbl_enc.classes_[target]})") 
pd.DataFrame(final_preds).to_csv("predictions_clip.csv", index=False)
print("Done")