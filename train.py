import torch
import torch.nn as nn
import torch.utils.data as d
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.tensorboard as tb

import time
import numpy as np
import datasets as data
import model as m
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import product

batches = [2000, 1000, 700, 500, 300, 100]
learings = [.0005, .0003, .0001, .00007, .00005, .00003, .00001]

# best 500, .0005

for LEARNING_RATE, BATCH_SIZE in tqdm(product(learings, batches)):
    # Hyperparameter definition
    MAX_EPOCHS = 12

    # Predisposes running on GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Data Initialisation
    train_data = data.ENG_WUsL(5)
    test_data = data.Test_ENG_WUsL()
    batch_generator = d.DataLoader(train_data, BATCH_SIZE, drop_last=True)
    test_generator = d.DataLoader(test_data, len(test_data), drop_last=False)
    print(len(batch_generator.dataset))
    in_shape = batch_generator.dataset.in_shape
    out_shape = batch_generator.dataset.out_shape

    # Model Initialisation
    criterion = nn.MSELoss()
    model = m.CLRM(in_shape, out_shape).to(device)
    optimiser = optim.Adagrad(model.parameters(),lr=LEARNING_RATE)

    # Tensorboard Initialisation
    writer = tb.SummaryWriter(log_dir=f"runs/{time.time()}_SC-IRM:lr={LEARNING_RATE} bs={BATCH_SIZE}")

    for epoch in range(MAX_EPOCHS):
        optimiser.zero_grad()

        epoch_loss = 0
        test_loss = 0

        for word, label in batch_generator:
            prediction = model(word).squeeze()
            # Calculates loss

            loss = criterion(prediction, label)

            epoch_loss += loss

            loss.backward()
            optimiser.step()

        writer.add_scalar("Loss", epoch_loss, epoch)
        writer.add_image("Prediction", prediction.unsqueeze(0), epoch)
        writer.add_image("Target", label.view(BATCH_SIZE, -1).unsqueeze(0), epoch)

        with torch.no_grad():
            optimiser.zero_grad()
            for test_word, test_label in test_generator:
                test_prediction = model(test_word).squeeze()
                # Calculates loss

                t_loss = criterion(test_prediction, test_label)

                test_loss += t_loss
                arr = test_prediction.cpu().numpy()
                np.savetxt(f"Output/ENG/Strat-{test_loss}.csv", arr)
            optimiser.zero_grad()
        
        writer.add_scalar("Test Loss", test_loss, epoch)
        writer.add_image("Test Prediction", test_prediction.unsqueeze(0), epoch)
        writer.add_image("Test Target", test_label.view(len(test_data), -1).unsqueeze(0), epoch)


        
    







        