import optuna

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from network import ContextUnet
from utils import CustomDataset, perturb_input

################################################################################

def train(nn_model, n_epoch, optimizer, lr, dataloader, device, ab_t, timesteps=500) -> list:
    nn_model.train()

    for ep in range(n_epoch):
        optimizer.param_groups[0]['lr'] = lr * (1-ep/n_epoch) # linearly decay learning rate (lr scheduler)
        total_loss = 0
        for x, _ in dataloader:   # x: images
            optimizer.zero_grad()
            x = x.to(device)
            
            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device) 
            x_pert = perturb_input(x, t, noise, ab_t)
            
            # use network to recover noise
            pred_noise = nn_model(x_pert, t / timesteps)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            total_loss += loss.item()
            
            optimizer.step()
    
    return total_loss

################################################################################

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

dataset = CustomDataset("/data/sprites.npy", "/data/sprites_labels.npy", transform, null_context=False)

def objective(trial):
    timesteps = trial.suggest_int('timestep', 100, 1000)
    beta1 = trial.suggest_float('beta1', 1e-4, 1e-3)
    beta2 = trial.suggest_float('beta2', 0.01, 0.1)
    n_feat = trial.suggest_int('n_feat', 32, 128)
    n_cfeat = trial.suggest_int('n_cfeat', 2, 10)
    lrate = trial.suggest_float('lrate', 1e-3, 1e-1)
    n_epoch = trial.suggest_int('n_epoch', 50, 200)
    batch_size = trial.suggest_int('batch_size', 128, 1024)
    
    height = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # construct DDPM noise scheduler
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    loss_hist = train(nn_model, n_epoch, optimizer, lrate, dataloader, device, ab_t, timesteps)

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open('study.txt', 'w') as f:
        f.write(str(study.best_trial))

if __name__ == "__main__":
    main()