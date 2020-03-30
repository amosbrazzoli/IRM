import torch
import torch.nn as nn
import torch.utils.data as d
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.tensorboard as tb

import time
import datasets as data
import model as m

from tqdm import tqdm
from itertools import product

batches = [1000, 700, 500, 300]
learings = [.0007, .0005, .0003]

# best 500, .0005

for LEARNING_RATE, BATCH_SIZE in product(learings, batches):
    # Hyperparameter definition
    MAX_EPOCHS = 50

    # Predisposes running on GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")



    # Data Initialisation
    batch_generator = d.DataLoader(data.ENG_WUsL(), BATCH_SIZE, drop_last=True)
    print(len(batch_generator.dataset))
    in_shape = batch_generator.dataset.in_shape
    out_shape = batch_generator.dataset.out_shape

    # Model Initialisation
    criterion = nn.MSELoss()
    model = m.CLRM(in_shape, out_shape).to(device)
    optimiser = optim.Adagrad(model.parameters(),lr=LEARNING_RATE)

    # Accuracy divisior
    entry_count = BATCH_SIZE * (len(batch_generator.dataset)//BATCH_SIZE) * 22 * 2
    print(entry_count)
    # Tensorboard Initialisation
    writer = tb.SummaryWriter(log_dir=f"runs/DC-IRM:lr={LEARNING_RATE} bs={BATCH_SIZE}")

    for epoch in range(MAX_EPOCHS):
        optimiser.zero_grad()

        epoch_loss = 0
        epoch_corrects = 0

        for word, label in tqdm(batch_generator):

            prediction = model(word)
        
            # Calculates loss

            #print(label.shape, prediction.shape)
            loss = criterion(label, prediction)

            epoch_loss += loss
            argmax_prediction = torch.argmax(prediction, dim = 1)
            argmax_label = torch.argmax(label, dim = 1)
            #print(argmax_label.shape, argmax_prediction.shape)
            epoch_corrects += int(torch.sum(torch.eq(argmax_prediction, argmax_label)))

            loss.backward()
            optimiser.step()
        item_accuracy = epoch_corrects / entry_count
        print(epoch_corrects, entry_count)


        writer.add_scalar("Loss", epoch_loss, epoch)
        writer.add_scalar("Entry Accuracy", item_accuracy, epoch)
        writer.add_image("Prediction", prediction.permute(1,0,2,3).view(2, BATCH_SIZE, -1), epoch)
        writer.add_image("Maxed_Prediction", argmax_prediction.view(BATCH_SIZE, -1).unsqueeze(0), epoch)
        writer.add_image("Target", argmax_label.view(BATCH_SIZE, -1).unsqueeze(0), epoch)

        print(f"Epoch: {epoch}\tLoss: {epoch_loss}\tItem Accuracy: {item_accuracy}")
        
    







        