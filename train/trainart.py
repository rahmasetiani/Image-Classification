import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.art_dataset import ArtDataset
from model.cnn_former import CNNFormer


class ArtClassifier:
    def __init__(self, art_a_path, art_b_path, feature_dim=3, device='cpu'):
        self.device = device
        self.model = CNNFormer(feature_dim=feature_dim).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()
        self.art_a = np.array(self.ambil_data(art_a_path))
        self.art_b = np.array(self.ambil_data(art_b_path))
        self.train_data = DataLoader(ArtDataset([
            (self.art_a[:11], 0),
            (self.art_b[:11], 1)
        ]), batch_size=8, shuffle=True)
        self.test_data = DataLoader(ArtDataset([
            (self.art_a[11:], 0),
            (self.art_b[11:], 1)
        ]), batch_size=8, shuffle=True)

    def ambil_data(self, folder):
        art = []
        dir_list = os.listdir(folder)
        for i in dir_list:
            data = cv2.imread(os.path.join(folder, i))
            data = cv2.resize(data, (300, 300))
            data = data / 255
            art.append(data)
        return art

    def train(self, epochs):
        loss_all = []
        for epoch in range(epochs):
            self.model.train()
            loss_total = 0
            for batch, (src, trg) in enumerate(self.train_data):
                src = src.permute(0, 3, 1, 2).to(self.device)
                pred = self.model(src)
                loss = self.criterion(pred, trg.to(self.device))
                loss_total += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            loss_batch = loss_total / len(self.train_data)
            loss_all.append(loss_batch)
            print(f'Epoch {epoch + 1}: Loss = {loss_batch:.4f}')
        plt.plot(loss_all, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()

    def evaluate(self, epochs):
        train_loss_all = []
        val_loss_all = []
        for epoch in range(epochs):
            self.model.train()
            train_loss_total = 0
            for batch, (src, trg) in enumerate(self.train_data):
                src = src.permute(0, 3, 1, 2).to(self.device)
                pred = self.model(src)
                loss = self.criterion(pred, trg.to(self.device))
                train_loss_total += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_loss_batch = train_loss_total / len(self.train_data)
            train_loss_all.append(train_loss_batch)

            self.model.eval()
            valid_loss_total = 0
            with torch.no_grad():
                for batch, (src, trg) in enumerate(self.test_data):
                    src = src.permute(0, 3, 1, 2).to(self.device)
                    pred = self.model(src)
                    loss = self.criterion(pred, trg.to(self.device))
                    valid_loss_total += loss.item()
            valid_loss_batch = valid_loss_total / len(self.test_data)
            val_loss_all.append(valid_loss_batch)
            print(f'Epoch {epoch + 1}: Training Loss = {train_loss_batch:.4f}, Validation Loss = {valid_loss_batch:.4f}')
        plt.plot(train_loss_all, label='Training Loss')
        plt.plot(val_loss_all, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self):
        self.model.eval()
        total_correct_predictions = 0
        total_test_samples = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch, (src, trg) in enumerate(self.test_data):
                src = src.permute(0, 3, 1, 2).to(self.device)
                trg = trg.to(self.device)
                pred = self.model(src)
                total_correct_predictions += (torch.argmax(pred, dim=1) == torch.argmax(trg, dim=1)).sum().item()
                total_test_samples += trg.size(0)
                all_preds.extend(torch.argmax(pred, dim=1).cpu().numpy())
                all_labels.extend(torch.argmax(trg, dim=1).cpu().numpy())
        accuracy = total_correct_predictions / total_test_samples
        print(f'Total Test Sample: {total_test_samples}')
        print(f'Total Test Correct: {total_correct_predictions}')
        print(f'Total Test Accuracy: {accuracy * 100:.2f}%')
        conf_matrix = confusion_matrix(all_labels, all_preds)
        class_names = ['Doberman', 'Maltippo']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    def compute_roc_auc(self):
        self.model.eval()
        y_true = []
        y_scores = []
        with torch.no_grad():
            for batch, (src, trg) in enumerate(self.test_data):
                src = src.permute(0, 3, 1, 2).to(self.device)
                trg = trg.to(self.device)
                pred = self.model(src)
                trg_binary = torch.argmax(trg, dim=1)
                pred_probs = F.softmax(pred, dim=1)
                pred_positive_probs = pred_probs[:, 1]
                y_true.extend(trg_binary.cpu().numpy())
                y_scores.extend(pred_positive_probs.cpu().numpy())
        roc_auc = roc_auc_score(y_true, y_scores)
        print(f'ROC AUC Score: {roc_auc:.4f}')
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()


