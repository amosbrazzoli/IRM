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

# Hyperparameter definition
BATCH_SIZE = 1000
LEARNING_RATE = 0.005
MAX_EPOCHS = 5

# Predisposes running on GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")



# Data Initialisation
batch_generator = d.DataLoader(data.ENG_WUsL(8), BATCH_SIZE)
print(f"Dataset Lenght:{len(batch_generator.dataset)}")
in_shape = batch_generator.dataset.in_shape
out_shape = batch_generator.dataset.out_shape

# Model Initialisation
criterion = nn.MSELoss(reduction="mean")
model = m.CLRM(in_shape, out_shape).to(device)
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Accuracy divisior
# gets the number of entries in a dataset 
one_column_entrywise_count = float(len(batch_generator.dataset)*batch_generator.dataset.out_shape[0])


# Tensorboard Initialisation
writer = tb.SummaryWriter(log_dir=f"runs/{time.time()}")



for epoch in range(MAX_EPOCHS):
    optimiser.zero_grad()

    epoch_loss = 0
    epoch_pron_corrects = 0
    epoch_decision_corrects = 0
    epoch_naming_corrects = 0
    c = 0

    for word, pron in tqdm(batch_generator):
        pron_hat = model(word)

        # Calculates loss
        loss = criterion(pron, pron_hat)
        epoch_loss += loss

        # Compiutes accuracy for naming time
        naming_time = pron[:, :, -1]
        predicted_naming = pron_hat[:, :, -1]
        naming_eq = torch.eq(naming_time, predicted_naming)
        epoch_naming_corrects += float(naming_eq.sum())

        # comptes accuracy for lexical decision
        decision_time = pron[:, :, -2]
        predicted_decision = pron_hat[:, :, -2]
        decision_eq = torch.eq(decision_time, predicted_decision)
        epoch_decision_corrects += float(decision_eq.sum())
        
        # computes accuracy for pronuntiation
        pron_form = pron[:, :, :-2]
        predicted_pron = pron_hat[:, :, :-2]
        pron_eq = torch.eq(pron_form ,predicted_pron)
        epoch_pron_corrects += float(pron_eq.sum())

        # backpropagates and steps the optimiser
        loss.backward()
        optimiser.step()
    
    naming_accuracy = (epoch_naming_corrects/one_column_entrywise_count)*100 
    decision_accuracy = (epoch_decision_corrects/one_column_entrywise_count)*100
    pronuntiation_accuracy = (epoch_pron_corrects/(one_column_entrywise_count*(batch_generator.dataset.out_shape[1]-2)))*100

    # For every epoch adds data to tensorboard
    writer.add_scalar("Loss", epoch_loss, epoch)
    writer.add_scalar("Naming Accuracy", naming_accuracy, epoch)
    writer.add_scalar("Decision Accuracy", decision_accuracy, epoch)
    writer.add_scalar("Pronuntiation Accuracy", pronuntiation_accuracy, epoch)
    writer.add_image("Prediciton", predicted_pron[0].unsqueeze(0), epoch)
    writer.add_image("Naming Prediction", naming_time.unsqueeze(0), epoch)
    writer.add_image("Decision Prediction", decision_time.unsqueeze(0), epoch)


    print(epoch_loss, epoch_pron_corrects, epoch_naming_corrects, epoch_decision_corrects)
    print(f"Epoch: {epoch}\tLoss: {epoch_loss}\tPronuntiation Accuracy: {pronuntiation_accuracy}\tNaming Accuracy: {naming_accuracy}\tDecision Accuracy: {decision_accuracy}")






        