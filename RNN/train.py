import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import RNNModel
from rnn_tokenize import tokenize, create_dataset
import numpy as np
import random

def train():
    batch_size = 1024
    n_iters = 100000
    visible_gpus = 0
    seed = 777
    # Create RNN
    input_dim = 512  # input dimension
    hidden_dim = 1024  # hidden layer dimension
    layer_dim = 5  # number of hidden layers
    output_dim = 2  # output dimension
    seq_len = 20

    device = "cpu" if visible_gpus == '-1' else f"cuda:{visible_gpus}"
    device_id = 0 if device == f"cuda" else -1

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    tokenized_data, embedding_model, _ = tokenize()
    input_list, input_train, input_test, target_train, target_test = create_dataset(tokenized_data, embedding_model, input_dim, seq_len)

    num_epochs = n_iters / (len(input_list) / batch_size)
    num_epochs = int(num_epochs)

    # Pytorch train and test sets
    input_tensor_train = torch.from_numpy(np.array(input_train, dtype=np.float64)).float()
    input_tensor_test = torch.from_numpy(np.array(input_test, dtype=np.float64)).float()

    target_tensor_train = torch.from_numpy(np.array(target_train, dtype=np.float64)).float().type(torch.LongTensor)
    target_tensor_test = torch.from_numpy(np.array(target_test, dtype=np.float64)).float().type(torch.LongTensor)

    # Pytorch train and test sets
    train = TensorDataset(input_tensor_train, target_tensor_train)
    test = TensorDataset(input_tensor_test, target_tensor_test)

    # data loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, device)

    # Cross Entropy Loss
    error = nn.CrossEntropyLoss()

    # SGD Optimizer
    learning_rate = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    seq_dim = 20
    loss_list = []
    iteration_list = []
    accuracy_list = []
    count = 0
    print("training start")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            train = Variable(images.view(-1, seq_dim, input_dim))
            labels = Variable(labels)

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(train)

            # Calculate softmax and ross entropy loss

            loss = error(outputs, labels.to(device=device))

            # Calculating gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            count += 1

            if count % 25 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    images = Variable(images.view(-1, seq_dim, input_dim))

                    # Forward propagation
                    outputs = model(images)

                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]

                    # Total number of labels
                    total += labels.size(0)

                    correct += (predicted == labels.to(device=device)).sum()

                accuracy = 100 * correct / float(total)

                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
                if count % 50 == 0:
                    # Print Loss
                    print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
                    if count % 1000 == 0:
                        torch.save(model.state_dict(), './model/model_' + str(count) + '.pth')

if __name__ == '__main__':
    train()