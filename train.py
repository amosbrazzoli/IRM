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
MAX_EPOCHS = 100
SHRINKER = 0.1

# Predisposes running on GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")



# Data Initialisation
batch_generator = d.DataLoader(data.ENG_WUsL(), BATCH_SIZE, drop_last=True)
print(len(batch_generator.dataset))
in_shape = batch_generator.dataset.in_shape
out_shape = batch_generator.dataset.out_shape

# Model Initialisation
criterion = nn.MSELoss(reduction="sum")
model = m.CLRM(in_shape, out_shape).to(device)
optimiser = optim.Adam(model.parameters())

# Accuracy divisior
one_column_entrywise_count = float(len(batch_generator.dataset)*batch_generator.dataset.out_shape[0])


# Tensorboard Initialisation
writer = tb.SummaryWriter(log_dir=f"runs/{time.time()}")

for epoch in range(MAX_EPOCHS):
    optimiser.zero_grad()

    epoch_loss = 0
    epoch_corrects = 0
    c = 0

    for word, pron in tqdm(batch_generator):
        #with torch.no_grad():
            #model.lin1.weight.data = F.hardshrink(model.lin1.weight.data, lambd=SHRINKER)
        pron_hat = model(word)

        # Calculates loss
        loss = criterion(pron, pron_hat)

        epoch_corrects += torch.sum(torch.eq(pron_hat, pron))
        c += torch.prod(torch.tensor(pron.shape))

        loss.backward()
        optimiser.step()
    
    item_accuracy = epoch_corrects/c

    print(pron.shape)
    print(pron_hat.shape)

    writer.add_scalar("Loss", epoch_loss, epoch)
    writer.add_scalar("Entry Accuracy", item_accuracy, epoch)
    writer.add_image("Prediction", pron_hat.unsqueeze(0).view(1, BATCH_SIZE, -1), epoch)
    writer.add_image("Target", pron.unsqueeze(0).view(1, BATCH_SIZE, -1), epoch)

    print(f"Epoch: {epoch}\tLoss: {epoch_loss}\tItem Accuracy: {item_accuracy}")
        
    







        