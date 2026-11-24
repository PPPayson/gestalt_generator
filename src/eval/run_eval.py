import argparse
import json
import os
import traceback

import numpy as np
import pandas as pd
import timm
import torch
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import IllusionDataset
from src.utils.eval_utils import get_basic_transform


def run_feature_comparison(args):
    device = torch.device(f'cuda:{args.gpu_id}')
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0, in_chans=1)
    
    data_config = timm.data.resolve_model_data_config(model)
    transform = get_basic_transform(data_config)

    train_dataset = IllusionDataset(transform=transform)
    # val_dataset = IllusionDataset()

    
    train_loader = DataLoader(train_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    # val_loader = DataLoader(val_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True)
    model = model.to(device)
    closure = feature_comparison(model, train_loader, device)

    return closure

def feature_comparison(model, data_loader, device):
    model.eval()
    all_closures = []
    with torch.inference_mode():
        for batch in data_loader:
            contour_batch, pos_batch, neg_batch = batch
            contour_batch = contour_batch.to(device)
            pos_batch = pos_batch.to(device)
            neg_batch = neg_batch.to(device)
            contour_features = model(contour_batch)
            pos_features = model(pos_batch)
            neg_features = model(neg_batch) # batch, d
            pos_contour = cosine_similarity(pos_features, contour_features)
            neg_contour = cosine_similarity(neg_features, contour_features)
            closure = torch.mean(pos_contour - neg_contour)
            all_closures.append(closure.item())
    return np.mean(all_closures)
    
def extract_features(model, data_loader, device, return_path = False):
    model.eval()
    features = []
    labels_list = []
    img_path_list = []
    for data in tqdm(data_loader):
        if return_path:
            images, labels, img_paths = data
            img_path_list+=(img_paths)
        else:
            images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(images)
            features.append(preds.cpu())
            labels_list.append(labels.cpu())
    features = torch.cat(features)
    labels = torch.cat(labels_list).squeeze()
    if not return_path:
        return features, labels
    else:
        return features, labels, img_path_list

def get_args_parser():
    parser = argparse.ArgumentParser("Linear Probe with Illusionary Contour Data")
    parser.add_argument('--model_name', type=str, default="vit_base_patch16_224.augreg_in21k_ft_in1k", help="TIMM model name")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--extract_batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', required=False, type=float, default=0.9)
    parser.add_argument('--num_workers', required=False, default=4, type=int)
    parser.add_argument('--gpu_id', required=False, type=int, default=0)
    parser.add_argument('--wandb', action='store_true', default=False)
    
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    output_file = "outs/closure_2.json"
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            output_dict = json.load(f)
    else:
        output_dict = {}
    timm_results_csv = os.path.join('assets', 'results-imagenet.csv')
    result_df = pd.read_csv(timm_results_csv)
    num_models = len(result_df.index)
    for index in reversed(range(num_models)):
        model_name = result_df['model'][index]
        if model_name in output_dict and output_dict[model_name] != "Failed":
            continue
        print(model_name)
        args.model_name = model_name
        try:
            closure = run_feature_comparison(args)
            output_dict[model_name] = closure
        except Exception as error:
            print(f"Failed for {model_name}: {error}")
            traceback.print_exc()
            output_dict[model_name] = "Failed"
        finally:
            with open(output_file, 'w') as f:
                output_json = json.dumps(output_dict, indent=4)
                f.write(output_json)