import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from utils import TimeSeriesFNO, load_and_proc_data, train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
directory = "data/XJTU_data"
file_list = csv_files = [directory+'/'+f for f in os.listdir(directory) if f.endswith(".csv")]
for f in file_list:
    print(f)
    
SEQ_LEN = 10
BATCH_SIZE = 128
features = ['voltage mean','voltage std','voltage kurtosis','voltage skewness','CC Q','CC charge time','voltage slope','voltage entropy','current mean','current std','current kurtosis','current skewness','CV Q','CV charge time','current slope','current entropy','capacity']
targets = ['capacity']
NUM_FEATURES = len(features)
NUM_TARGETS = len(targets)

_, _, train_loader, val_loader, test_loader, scaler_data = load_and_proc_data(file_list,
                                                                              features = features,
                                                                              targets=targets,
                                                                              SEQ_LEN = SEQ_LEN,
                                                                              BATCH_SIZE = BATCH_SIZE)


# Model
model = TimeSeriesFNO(NUM_FEATURES, NUM_TARGETS, SEQ_LEN, SEQ_LEN)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

train_model(model, train_loader, val_loader, loss_fn, optimizer, "models/FNO_model.pth", device, num_epochs=10)

evaluate_model(model, test_loader, "models/FNO_model.pth", 'outputs/error_results_FNO.txt', 'FNO', plot_fig = True, device=device)
