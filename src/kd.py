import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from network import ContextUnet
from utils import CustomDataset, perturb_input, denoise_add_noise, plot_sample, plot_grid, knowledge_distillation_loss, get_trainable_params, set_logger

################################################################################

def train(nn_model, n_epoch, optimizer, lr, dataloader, device, ab_t, criterion, timesteps=500) -> list:
    nn_model.train()
    loss_hist = []

    with tqdm(range(n_epoch), desc="Training") as pbar:
        for ep in pbar:
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
                loss = criterion(pred_noise, noise)
                loss.backward()
                total_loss += loss.item()
                
                optimizer.step()

            pbar.set_postfix({'loss': total_loss, 'lr': optimizer.param_groups[0]['lr']})
            loss_hist.append(total_loss)
    
    return loss_hist

def trainStudent(teacher_model, student_model, n_epoch, optimizer, lr, dataloader, device, ab_t, timesteps=500) -> list:
    teacher_model.eval()
    student_model.train()
    loss_hist = []

    with tqdm(range(n_epoch), desc="Training") as pbar:
        for ep in pbar:
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
                teacher_noise = teacher_model(x_pert, t / timesteps)
                student_noise = student_model(x_pert, t / timesteps)
                loss = knowledge_distillation_loss(teacher_noise, student_noise, noise)
                loss.backward()
                total_loss += loss.item()

                optimizer.step()

            pbar.set_postfix({'loss': total_loss, 'lr': optimizer.param_groups[0]['lr']})
            loss_hist.append(total_loss)
    
    return loss_hist

################################################################################

@torch.no_grad()
def sample_ddpm(nn_model, n_sample, height, timesteps, device, ab_t, a_t, b_t, save_rate=20):
    nn_model.eval()
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t)    # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, ab_t, a_t, b_t, z)
        if i % save_rate ==0 or i==timesteps or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

################################################################################

def main():
    set_logger(f"{time()}.log")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # diffusion hyperparameters
    timesteps = 500
    beta1 = 1e-4
    beta2 = 0.02

    # construct DDPM noise scheduler
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
    ab_t[0] = 1

    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    logging.info(f"Using device: {device}")
    n_feat_teacher = 64 # 64 hidden dimension feature
    n_feat_student = 32 # 32 hidden dimension feature
    n_cfeat = 5 # context vector is of size 5
    height = 16 # 16x16 image
    save_dir = './results/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # training hyperparameters
    batch_size = 1280
    n_epoch = 200
    lrate = 1e-2

    # load dataset
    logging.info("Loading dataset...")
    dataset = CustomDataset("/data/sprites.npy", "/data/sprites_labels.npy", transform, null_context=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # initialize teacher model
    teacher_model = ContextUnet(in_channels=3, n_feat=n_feat_teacher, n_cfeat=n_cfeat, height=height).to(device)
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=lrate)
    criterion = torch.nn.MSELoss()
    logging.info(f"Number of trainable parameters in teacher model: {get_trainable_params(teacher_model)}")

    # train the model
    teacher_loss_hist = train(teacher_model, n_epoch, optimizer, lrate, dataloader, device, ab_t, criterion, timesteps)

    # save the model
    if not os.path.exists('./models'): os.makedirs('./models')
    torch.save(teacher_model.state_dict(), './models/model.pth')
    pickle.dump(ab_t, open('./models/ab_t.pkl', 'wb'))
    pickle.dump(a_t, open('./models/a_t.pkl', 'wb'))
    pickle.dump(b_t, open('./models/b_t.pkl', 'wb'))
    logging.info("Teacher Model saved.")

    # teacher_model = ContextUnet(in_channels=3, n_feat=n_feat_teacher, n_cfeat=n_cfeat, height=height).to(device)
    # teacher_model.load_state_dict(torch.load('./models/model.pth'))

    # initialize student model
    student_model = ContextUnet(in_channels=3, n_feat=n_feat_student, n_cfeat=5, height=16).to(device)
    student_optimizer = torch.optim.Adam(student_model.parameters(), lr=lrate)
    logging.info(f"Number of trainable parameters in student model: {get_trainable_params(student_model)}")

    # train the student model
    student_loss_hist = trainStudent(teacher_model, student_model, n_epoch, student_optimizer, lrate, dataloader, device, ab_t, timesteps)

    # save the student model
    torch.save(student_model.state_dict(), './models/student_model.pth')
    logging.info("Student Model saved.")

    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(teacher_loss_hist, label='teacher')
    plt.plot(student_loss_hist, label='student')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()
    logging.info("Training Loss plot saved.")

################################################################################

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    nn_model = ContextUnet(in_channels=3, n_feat=32, n_cfeat=5, height=16).to(device)
    nn_model.load_state_dict(torch.load('./models/student_model.pth'))
    height = 16
    timesteps = 500
    save_dir = './results/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # load diffusion hyperparameters
    ab_t = pickle.load(open('./models/ab_t.pkl', 'rb'))
    a_t = pickle.load(open('./models/a_t.pkl', 'rb'))
    b_t = pickle.load(open('./models/b_t.pkl', 'rb'))

    # sample from the model
    n_sample = 32
    samples, intermediate = sample_ddpm(nn_model, n_sample, height, timesteps, device, ab_t, a_t, b_t, save_rate=20)
    animation_ddpm = plot_sample(intermediate, 32, 4, save_dir, f"{time()}", None, save=True)
    plot_grid(samples, n_sample, 4, save_dir, timesteps)

################################################################################


if __name__ == "__main__":
    main()

    test()