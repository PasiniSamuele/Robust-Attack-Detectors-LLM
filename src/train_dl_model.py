from utils.path_utils import get_last_run_number
import os
import pandas as pd
from utils.utils import init_argument_parser
import torch
from detection_models.utils.general import process_payloads
from detection_models.datasets.xss_dataset import XSSDataset
from detection_models.architectures.CNN import CNNDetector
from detection_models.architectures.MLP import MLPDetector
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def train_epoch(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for i, (payloads, labels) in enumerate(train_loader):
        payloads = payloads.to(device)
        labels = labels.to(torch.float32).to(device)
        optimizer.zero_grad()
        outputs = model(payloads)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def val_epoch(val_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for i, (payloads, labels) in enumerate(val_loader):
            payloads = payloads.to(device)
            labels = labels.to(torch.float32).to(device)
            outputs = model(payloads)
            predicted = torch.round(outputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        accuracy = 100.0 * n_correct / n_samples
    return total_loss / len(val_loader), accuracy

def train(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = pd.read_csv(opt.trainset).sample(frac=1)
    sorted_tokens, train_cleaned_tokenized_payloads = process_payloads(train_set)
    train_class_labels = train_set['Class']

    # Get the vocab size
    vocab_size = len(sorted_tokens)

    # Create a dataset and dataloader
    train_dataset = XSSDataset(train_cleaned_tokenized_payloads, train_class_labels)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)

    # Validation set
    validation_set = pd.read_csv(opt.valset).sample(frac=1)
    validation_cleaned_tokenized_payloads = process_payloads(validation_set)[1]
    validation_class_labels = validation_set['Class']
    validation_dataset = XSSDataset(validation_cleaned_tokenized_payloads, validation_class_labels)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=opt.batch_size, shuffle=False)

    if opt.model == 'mlp':
        model_architecture = MLPDetector
    elif opt.model == 'cnn':
        model_architecture = CNNDetector
    
    model = model_architecture(vocab_size, opt.embedding_dim, XSSDataset.MAX_LENGTH).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)

    runs_folder = os.path.join(opt.runs_folder, opt.model)
    os.makedirs(runs_folder, exist_ok=True)
    last_run = get_last_run_number(runs_folder)
    runs_folder = os.path.join(runs_folder, f"run_{last_run + 1}")
    os.makedirs(runs_folder, exist_ok=True)
    writer = SummaryWriter(runs_folder)
    epochs_without_improvement = 0

    for epoch in range(opt.epochs):
        train_loss = train_epoch(train_loader, model, criterion, optimizer, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        val_loss, val_accuracy = val_epoch(validation_loader, model, criterion, device)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        print(f"Epoch {epoch} - Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

        if epoch == 0 or val_loss < best_val_loss:
            epochs_without_improvement = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(runs_folder, 'checkpoint.pth'))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > opt.patience:
                print("Early stopping")
                break

            

def add_parse_arguments(parser):

    parser.add_argument('--trainset', type=str, required=True, help='Training dataset')
    parser.add_argument('--valset', type=str, required=True, help='Validation dataset')
    parser.add_argument('--model', type=str, default='mlp', help='mlp | cnn')
    parser.add_argument('--runs_folder', type=str, default="src/detection_models/runs", help='Runs Folder')

    #hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--embedding_dim', type=float, default=8, help='size of the embeddings')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')

    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    train(opt)

if __name__ == '__main__':
    main()