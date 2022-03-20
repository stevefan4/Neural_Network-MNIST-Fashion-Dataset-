############################################################
# CIS 521: Neural Network for Fashion MNIST Dataset
############################################################

student_name = "Steven Fandozzi"

############################################################
# Imports
############################################################

import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import torch.nn as nn
import torch.nn.functional as F



# Include your imports here, if any are used.
import csv

############################################################
# Neural Networks
############################################################
def load_data(file_path, reshape_images):
    loaded_data = []
    labels = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not(reshape_images):
                labels.append(row['label'])
                del row['label']
                hold = np.array(list(row.values())).astype(int)
                loaded_data.append(hold)
            else:
                labels.append(row['label'])
                del row['label'] 
                hold = np.array(list(row.values())).astype(int)
                for i in range(0, len(hold)):
                    hold[i] = int(hold[i])
                reshaped = np.reshape(hold, (1, 28, 28))
                loaded_data.append(reshaped)
    return np.array(loaded_data), np.array(labels).astype(int)

# PART 2.2
class EasyModel(torch.nn.Module):
    def __init__(self):
        super(EasyModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        #x = torch.tanh(self.fc(x))
        #x = F.relu(self.fc(x))
        #return F.log_softmax(x, dim=1)
        return self.fc(x)

# PART 2.3
class MediumModel(torch.nn.Module):
    def __init__(self):
        super(MediumModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# PART 2.4
class AdvancedModel(torch.nn.Module):
    def __init__(self):
        super(AdvancedModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.classifier(x)
        return x

############################################################
# Fashion MNIST dataset
############################################################

class FashionMNISTDataset(Dataset):
    def __init__(self, file_path, reshape_images):
        self.X, self.Y = load_data(file_path, reshape_images)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


############################################################
# Reference Code
############################################################

def train(model, data_loader, num_epochs, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loader):
            images = torch.autograd.Variable(images.float())
            labels = torch.autograd.Variable(labels)

            optimizer.zero_grad()
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                y_true, y_predicted = evaluate(model, data_loader)
                print(f'Epoch : {epoch}/{num_epochs}, '
                      f'Iteration : {i}/{len(data_loader)},  '
                      f'Loss: {loss.item():.4f},',
                      f'Train Accuracy: {100.* accuracy_score(y_true, y_predicted):.4f},',
                      f'Train F1 Score: {100.* f1_score(y_true, y_predicted, average="weighted"):.4f}')

def evaluate(model, data_loader):
    model.eval()
    y_true = []
    y_predicted = []
    for images, labels in data_loader:
        images = torch.autograd.Variable(images.float())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels)
        y_predicted.extend(predicted)
    return y_true, y_predicted

def plot_confusion_matrix(cm, class_names, title=None):
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



def main():
    class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001
    file_path = 'dataset.csv'

    data_loader = torch.utils.data.DataLoader(dataset=FashionMNISTDataset(file_path, False),
                                              batch_size=batch_size,
                                              shuffle=True)
    data_loader_reshaped = torch.utils.data.DataLoader(dataset=FashionMNISTDataset(file_path, True),
                                                       batch_size=batch_size,
                                                       shuffle=True)

    # EASY MODEL
    easy_model = EasyModel()
    train(easy_model, data_loader, num_epochs, learning_rate)
    y_true_easy, y_pred_easy = evaluate(easy_model, data_loader)
    print(f'Easy Model: '
          f'Final Train Accuracy: {100.* accuracy_score(y_true_easy, y_pred_easy):.4f},',
          f'Final Train F1 Score: {100.* f1_score(y_true_easy, y_pred_easy, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_easy, y_pred_easy), class_names, 'Easy Model')

    # MEDIUM MODEL
    medium_model = MediumModel()
    train(medium_model, data_loader, num_epochs, learning_rate)
    y_true_medium, y_pred_medium = evaluate(medium_model, data_loader)
    print(f'Medium Model: '
          f'Final Train Accuracy: {100.* accuracy_score(y_true_medium, y_pred_medium):.4f},',
          f'Final F1 Score: {100.* f1_score(y_true_medium, y_pred_medium, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_medium, y_pred_medium), class_names, 'Medium Model')

    # ADVANCED MODEL
    advanced_model = AdvancedModel()
    train(advanced_model, data_loader_reshaped, num_epochs, learning_rate)
    y_true_advanced, y_pred_advanced = evaluate(advanced_model, data_loader_reshaped)
    print(f'Advanced Model: '
          f'Final Train Accuracy: {100.* accuracy_score(y_true_advanced, y_pred_advanced):.4f},',
          f'Final F1 Score: {100.* f1_score(y_true_advanced, y_pred_advanced, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_advanced, y_pred_advanced), class_names, 'Advanced Model')

############################################################
# Feedback
############################################################

feedback_question_1 = """
It is clear that the easy-model confused the majority of classes for shirts. 
Looking at the medium and advanced models, it seems that classifying pullovers
as coats is a common misclassification, which seems reasonable. 
"""

feedback_question_2 = """
My advanced model does 2 convolutional layers with a kernel size of 3 and
utilize RELU activation functions and max pooling layers. There are also
2 Linear layers with initial dropout.
"""

feedback_question_3 = 6

feedback_question_4 = """
Most challenging part of this assignment for myself was properly loading the data in a 
usable tensor format. 
"""

feedback_question_5 = """
My favorite part of this assignment was building the three different models and 
seeing how confusion matrices compared. 
"""

if __name__ == '__main__':
    main()
