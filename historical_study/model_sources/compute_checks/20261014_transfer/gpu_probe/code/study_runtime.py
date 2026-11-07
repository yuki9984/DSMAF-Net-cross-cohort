"""Small runtime for the fixed study recipe, without the historical omnibus trainer.
The transform order, loss, clipping and metric definitions follow the executed
protocol. Optional legacy training modes are deliberately unsupported.
"""
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
 precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score)

@dataclass
class DatasetBundle:
    dataframe: object
    img_paths: object
    labels: object
    meta_array: object

class meta_img_dataset(Dataset):
    def __init__(self,img_path,meta_data,labels,transform=None):
        self.img_path,self.meta_data,self.labels,self.transform=img_path,meta_data,labels,transform
    def __len__(self):return len(self.labels)
    def __getitem__(self,index):
        with Image.open(self.img_path[index]) as source:
            rgb=source.convert("RGB")
        image=self.transform(rgb) if self.transform is not None else rgb
        return image,torch.from_numpy(self.meta_data[index]).float(),self.labels[index]
meta_img_dataset_test=meta_img_dataset

def set_seed(seed,fast_mode=False):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=not fast_mode
    torch.backends.cudnn.benchmark=fast_mode

def build_transforms(input_size=224):
    size=(int(input_size*1.15)//2)*2
    norm=lambda:transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    training=transforms.Compose([
        transforms.Resize((size,size)),transforms.RandomResizedCrop(input_size,scale=(.9,1.0)),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.ColorJitter(brightness=.05,contrast=.05,saturation=.05,hue=.01),
        transforms.RandomRotation(degrees=10),transforms.ToTensor(),norm()])
    evaluation=transforms.Compose([
        transforms.Resize((size,size)),transforms.CenterCrop(input_size),
        transforms.ToTensor(),norm()])
    return training,evaluation

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self,class_counts):
        super().__init__()
        self.register_buffer("log_class_prior",torch.as_tensor(class_counts,dtype=torch.float32).clamp(min=1).log())
    def forward(self,inputs,targets):
        return nn.functional.cross_entropy(inputs+self.log_class_prior.to(inputs.device),targets)

def _logits(value):return value["logits"] if isinstance(value,dict) else value
def _amp(enabled):
    # Match the CUDA AMP context of the executed recipe.
    return torch.amp.autocast("cuda",enabled=enabled)

def train_one_epoch(model,loader,criterion,optimizer,device,scaler=None,use_amp=False,
                    epoch=0,total_epochs=0,show_progress=True,ema_model=None,
                    mixup_alpha=0.,margin_lambda=0.,margin=.5,
                    cluster_aux_criterion=None,cluster_aux_lambda=0.):
    if ema_model is not None or mixup_alpha or margin_lambda or cluster_aux_criterion is not None or cluster_aux_lambda:
        raise ValueError("This release supports only the fixed study recipe; legacy optional modes are excluded")
    if use_amp and scaler is None:raise ValueError("AMP requires GradScaler")
    model.train();loss_total=0.;truth=[];predicted=[]
    for images,metadata,labels in loader:
        images=images.to(device,non_blocking=True)
        metadata=metadata.to(device,non_blocking=True)
        labels=labels.long().to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _amp(use_amp):
            logits=_logits(model(images,metadata));loss=criterion(logits,labels)
        if use_amp:
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
            scaler.step(optimizer);scaler.update()
        else:
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
            optimizer.step()
        loss_total+=loss.item()*images.size(0)
        truth.append(labels.cpu().numpy())
        predicted.append(torch.softmax(logits.detach(),dim=1).argmax(dim=1).cpu().numpy())
    return loss_total/len(loader.dataset),float(accuracy_score(np.concatenate(truth),np.concatenate(predicted)))

def _metrics(labels,logits,n):
    z=logits-np.max(logits,axis=1,keepdims=True)
    e=np.exp(z);prob=e/np.clip(e.sum(axis=1,keepdims=True),1e-12,None)
    pred=prob.argmax(axis=1)
    accuracy=float(accuracy_score(labels,pred))
    bacc=float(balanced_accuracy_score(labels,pred))
    weighted=float(f1_score(labels,pred,average="weighted",zero_division=0))
    cm=confusion_matrix(labels,pred,labels=list(range(n)))
    pr,re,f1,support=precision_recall_fscore_support(labels,pred,labels=list(range(n)),zero_division=0)
    sensitivities=[];specificities=[]
    for k in range(n):
        tp=cm[k,k];fn=cm[k,:].sum()-tp;fp=cm[:,k].sum()-tp;tn=cm.sum()-tp-fn-fp
        sensitivities.append(tp/(tp+fn+1e-8));specificities.append(tn/(tn+fp+1e-8))
    result=dict(acc=accuracy,accuracy=accuracy,bac=bacc,balanced_accuracy=bacc,bacc=bacc,
                precision=float(precision_score(labels,pred,average="weighted",zero_division=0)),
                f1=weighted,macro_f1=float(f1_score(labels,pred,average="macro",zero_division=0)),
                weighted_f1=weighted,sens=float(np.mean(sensitivities)),spec=float(np.mean(specificities)),
                confusion_matrix=cm.tolist(),per_class={})
    names=("ACK","BCC","MEL","NEV","SCC","SEK")
    for k in range(n):
        result["per_class"][names[k] if k<len(names) else str(k)]={"precision":float(pr[k]),"recall":float(re[k]),"f1":float(f1[k]),"support":int(support[k])}
    try:auc=float(roc_auc_score(np.eye(n)[labels],prob,average="macro",multi_class="ovr"))
    except Exception:auc=None
    result.update(auc=auc,macro_auc=auc)
    return result,prob,pred

@torch.no_grad()
def evaluate(model,loader,criterion,device,bundle,num_classes,split_name,
             save_prediction_path=None,use_amp=False,tta_horizontal_flip=False,
             epoch=0,total_epochs=0,show_progress=True):
    model.eval();truth=[];values=[];loss_total=0.
    for images,metadata,labels in loader:
        images=images.to(device,non_blocking=True)
        metadata=metadata.to(device,non_blocking=True)
        labels=labels.long().to(device,non_blocking=True)
        with _amp(use_amp):
            logits=_logits(model(images,metadata))
            if tta_horizontal_flip:logits=.5*(logits+_logits(model(torch.flip(images,dims=[3]),metadata)))
            loss=criterion(logits,labels) if criterion is not None else None
        if loss is not None:loss_total+=loss.item()*images.size(0)
        truth.append(labels.cpu().numpy());values.append(logits.cpu().numpy())
    y=np.concatenate(truth);z=np.concatenate(values).astype(np.float32)
    result,prob,pred=_metrics(y,z,num_classes)
    result.update(loss=loss_total/len(loader.dataset) if criterion is not None else None,split=split_name)
    if save_prediction_path is not None:
        frame=pd.DataFrame({"img_id":bundle.dataframe["img_id"].tolist(),"img_path":bundle.img_paths,"label":y,"pred":pred})
        for k in range(num_classes):frame[f"prob_{k}"]=prob[:,k];frame[f"logit_{k}"]=z[:,k]
        frame.to_csv(save_prediction_path,index=False)
    return result

