import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from codes import dataset, motion_encoder

import yaml
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import tqdm
import logging
from time import gmtime, strftime
from shutil import copy
import os


#TODO: Plot all the test set in a subplot with labels
#TODO: Do training for other datasets
#TODO: Delete this later
import warnings
warnings.filterwarnings('ignore')

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
          default='./configs/ckplus.yml', type=str)
    parser.add_argument('--log_dir', dest='log_dir', help='Path to logs and saved checkpoints.',
          default='./logs/', type=str)
    
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
    dataset_train = dataset.CKPlusDataset(**configs['data_params'],train=True, device='cpu') # device in dataset class should be cpu otherwise Cannot re-initialize CUDA in forked subprocess
    dataset_test = dataset.CKPlusDataset(**configs['data_params'], train=False, device='cpu')

    # Dataloaders
    train_loader = DataLoader(dataset_train, batch_size=configs['train_params']['batch_size'], shuffle=True, num_workers=configs['train_params']['num_workers'])
    test_loader = DataLoader(dataset_test, batch_size=configs['train_params']['batch_size'], shuffle=False, num_workers=configs['train_params']['num_workers'])

    #Model
    model = motion_encoder.ResNet18(in_channels=configs['images']['in_channels'], out_classes=configs['data_params']['out_class'], pretrained=True)
    model.to(device)

    # # Test Model and DLoader
    # for some_batch in tqdm.tqdm(train_loader):
    #     pass
    # print("TrainLoader OK")
    # for some_batch in tqdm.tqdm(test_loader):
    #     pass
    # print("TestLoader OK")
    # imgs, labels = next(iter(test_loader))

    # _ = model(imgs.to(device))
    # print(imgs.size(), labels.size())
    # print("Model Tested and ready")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = Adam(model.parameters(), lr=configs['train_params']['lr'])

    # MultiStepLR
    scheduler = lr_scheduler.MultiStepLR(optimizer, configs['train_params']['epoch_milestones'])

    # Training Loop
    writer = SummaryWriter(log_dir=log_dir)
    train_loss=[]
    for epoch in tqdm.trange(configs['train_params']['num_epochs']):
        for batch in tqdm.tqdm(train_loader):
            model.train()
            optimizer.zero_grad()

            img, labels = batch
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
            sum_accuracy=[]
            for i, batch in enumerate(test_loader):
                img, labels = batch
                img = img.to(device)
                labels = labels.to(device)
                logits = model(img)
                eval_loss.append(criterion(logits, labels).item())
                sum_accuracy.append(cal_accuracy(logits, labels))
            
            # Plot the image and pose using the 'draw_axis' function
            idx = 0
            img_toplot = torch.clone(img[idx, ...])
            predicted_emotion = dataset.EMOTIONS_CKPlus[torch.argmax(logits[idx].detach().cpu())]
            true_emotion = dataset.EMOTIONS_CKPlus[labels[idx]]

            # Add the image to TensorBoard
            writer.add_image(f"Predicted/True: {predicted_emotion}/{true_emotion}", img_toplot, global_step=epoch)
        
        # Logging the evaluation loss to TensorBoard
        avg_eval_loss = sum(eval_loss) / len(eval_loss)
        writer.add_scalar("Eval Loss", avg_eval_loss, epoch)

        # Logg the average accuracy of test set
        logg_accuracy(f"epoch: {epoch} - acc: {sum_accuracy}", os.path.join(log_dir, "test-set-acc.txt"))

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