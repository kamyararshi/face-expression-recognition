import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from codes import dataset, motion_encoder

import yaml
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from argparse import ArgumentParser
import tqdm
import logging
from time import gmtime, strftime
from shutil import copy
import os
import signal


import warnings
warnings.filterwarnings('ignore')

#TODO: Use it
def save_weights_on_interrupt(model, epoch, optimizer, train_loss, log_dir):
    """
    Save the model's weights to a file when the program is interrupted (e.g., CTRL+C).

    Args:
        model (torch.nn.Module): The PyTorch model whose weights you want to save.
        save_path (str): The path to save the model's weights.
    """
    def save_weights(signum, frame):
        print("Received interrupt signal. Saving model weights...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss[-1],
            }, os.path.join(log_dir, f'{epoch+1}.pth.tar'))
        
        print("Model weights saved.")
        exit(1)  # You can exit the program or perform other actions as needed

    # Register the signal handler
    signal.signal(signal.SIGINT, save_weights)

    # Start the training loop
    try:
        while True:
            # Your training loop code goes here
            # You can replace this with your actual training code
            pass
    except KeyboardInterrupt:
        pass  # Handle the interrupt signal with the signal handler

def cal_accuracy(predictions, true_labels, logits=True, in_precent=True):
    """Calculates the prediction accuracy
    """
    # Get the predicted labels by finding the index of the maximum value along
    # the second dimension (axis 1).
    if logits:
        predicted_labels = torch.argmax(predictions, dim=1)
    else:
        predicted_labels = predictions
    
    # Calculate the accuracy by comparing the predicted labels to the true labels.
    correct_predictions = (predicted_labels == true_labels).sum().item()
    total_examples = true_labels.size(0)
    accuracy = correct_predictions / total_examples

    return accuracy*100 if in_precent else accuracy

def logg_accuracy(input_string, path):
    """
    Write the given string to a text file. If the file does not exist, create it.

    Args:
    - filename: The path of the text file to write to.
    - input_string: The string to be written to the file.
    """
    try:
        # Open the file in 'a' (append) mode, which will create the file if it doesn't exist.
        with open(path, 'a') as file:
            # Write the input string followed by a newline character to a new line.
            file.write(input_string + '\n')
    except IOError as e:
        print(f"Error writing to file {path}: {e}")


def main():
    parser = ArgumentParser(description='Head pose estimation using the ResNet-18 network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--config', dest='config', help='Path to config file.',
          default='./configs/emotic.yml', type=str)
    parser.add_argument('--log_dir', dest='log_dir', help='Path to logs and saved checkpoints.',
          default='./logs/', type=str)
    parser.add_argument('--test_script', action=argparse.BooleanOptionalAction, help="tests the model, Dloader, and all the batches",
                        default=False, type=bool)
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, help="whether the backbone uses pretrained weights",
                        default=False, type=bool)
    
    args = parser.parse_args()

    device = (f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    with open(args.config) as f:
        configs = yaml.safe_load(f)

    # Log Dir
    log_dir = os.path.join(args.log_dir, os.path.basename(args.config).split('.')[0])
    log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime()) + '_e:' + str(configs["train_params"]["num_epochs"]) + '_lr:' + str(configs["train_params"]["lr"])
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
        copy(args.config, log_dir)

    
    # Dataset
    train_df_dir = os.path.join(configs['data_params']['root_dir'], "emotic_pre", "train.csv")
    val_df_dir = os.path.join(configs['data_params']['root_dir'], "emotic_pre", "val.csv")
    test_df_dir = os.path.join(configs['data_params']['root_dir'], "emotic_pre", "test.csv")

    
    # Mean and std of data for normalization
    context_norm = [configs['images']['context_mean'], configs['images']['context_std']]
    body_norm = [configs['images']['body_mean'], configs['images']['body_std']]

    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(dataset.EMOTIONS_Emotic):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion

    dataset_train = dataset.Emotic_CSVDataset(pd.read_csv(train_df_dir), cat2ind, context_norm, body_norm)
    dataset_test = dataset.Emotic_CSVDataset(pd.read_csv(test_df_dir), cat2ind, context_norm, body_norm)
    num_test_data = dataset_test.__len__()

    # Dataloaders
    train_loader = DataLoader(dataset_train, batch_size=configs['train_params']['batch_size'], shuffle=True, num_workers=configs['train_params']['num_workers'])
    test_loader = DataLoader(dataset_test, batch_size=configs['train_params']['batch_size'], shuffle=False, num_workers=configs['train_params']['num_workers'])

    #Model
    model = motion_encoder.ResNet18(in_channels=configs['images']['in_channels'], out_classes=configs['data_params']['out_class'], pretrained=args.pretrained)
    model.to(device)
    if args.pretrained: print("Model with pretrained weghts loaded")

    if args.test_script:
        # Test Model and DLoader
        for some_batch in tqdm.tqdm(train_loader):
            pass
        print("TrainLoader OK")
        for some_batch in tqdm.tqdm(test_loader):
            pass
        print("TestLoader OK")
        imgs, labels = next(iter(test_loader))

        _ = model(imgs.to(device))
        print(imgs.size(), labels.size())
        print("Model Tested and ready")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = Adam(model.parameters(), lr=configs['train_params']['lr'])

    # MultiStepLR
    scheduler = lr_scheduler.MultiStepLR(optimizer, configs['train_params']['epoch_milestones'])

    # Training Loop
    writer = SummaryWriter(log_dir=log_dir)
    train_loss=[]
    for epoch in tqdm.trange(configs['train_params']['num_epochs']):
        for batch in tqdm.tqdm(test_loader): #TODO: This is only trial to see what's wrong, it's training on the test set (smaller)
            model.train()
            optimizer.zero_grad()

            _, img, _, labels = batch
            img = img.to(device)
            labels = labels.to(device)
            logits = model(img)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
      
        train_loss.append(loss.detach().item())
        writer.add_scalar("Train Loss", loss.detach().item(), epoch)


        # Eval
        model.eval()
        eval_loss=[]
        with torch.no_grad():
            i=0
            sum_accuracy=0
            for i, batch in enumerate(test_loader):
                _, img, _, labels = batch
                img = img.to(device)
                labels = labels.to(device)
                logits = model(img)
                eval_loss.append(criterion(logits, labels).item())
                sum_accuracy += cal_accuracy(logits, labels) * img.shape[0]
            
            # Plot the image and label
            idx = 0
            img_toplot = torch.clone(img[idx, ...])
            predicted_emotion = dataset.EMOTIONS_CKPlus[torch.argmax(logits[idx].detach().cpu())-1]
            true_emotion = dataset.EMOTIONS_CKPlus[labels[idx]]

            # Add the image to TensorBoard
            writer.add_image(f"Predicted/True: {predicted_emotion}/{true_emotion}", img_toplot, global_step=epoch)
        
        # Logging the evaluation loss to TensorBoard
        avg_eval_loss = sum(eval_loss) / len(eval_loss)
        writer.add_scalar("Eval Loss", avg_eval_loss, epoch)

        # Logg the average accuracy of test set
        logg_accuracy(f"epoch: {epoch} - avg_acc: {(sum_accuracy/num_test_data):.2f} - loss: {eval_loss[-1]:.2f}", os.path.join(log_dir, "test-set-acc.txt"))

        scheduler.step()

    writer.close()

    # Save model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss[-1],
            }, os.path.join(log_dir, f'{epoch+1}.pth.tar'))


if __name__ == '__main__':
    
    main()