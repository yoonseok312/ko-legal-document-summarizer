import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import RNNModel
from LSTM import LSTMModel
from rnn_tokenize import tokenize, create_dataset
import numpy as np
import random
import pickle

def train():
    batch_size = 32
    n_iters = 20000
    visible_gpus = 0
    seed = 777
    # Create RNN
    input_dim = 128  # input dimension
    hidden_dim = 256  # hidden layer dimension
    layer_dim = 1  # number of hidden layers
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

    tokenized_data, tokenized_valid_data, embedding_model, train_data, valid_data, _ = tokenize(input_dim)
    input_train, input_test, target_train, target_test = create_dataset(
        tokenized_data,
        tokenized_valid_data,
        embedding_model,
        input_dim,
        seq_len,
        train_data,
        valid_data
    )

    with open("./data/valid_ext_list_hit", "rb") as f:
        valid_ext_list_hit = pickle.load(f)

    num_epochs = n_iters / (len(input_train) / batch_size)
    num_epochs = int(num_epochs)

    # Pytorch train and test sets
    input_tensor_train = torch.from_numpy(np.array(input_train, dtype=np.float64)).float()
    input_tensor_test = torch.from_numpy(np.array(input_test, dtype=np.float64)).float()

    target_tensor_train = torch.from_numpy(np.array(target_train, dtype=np.float64)).float().type(torch.LongTensor)
    target_tensor_test = torch.from_numpy(np.array(target_test, dtype=np.float64)).float().type(torch.LongTensor)

    # print(input_tensor_train.shape, target_tensor_train.shape)

    # Pytorch train and test sets
    train = TensorDataset(input_tensor_train, target_tensor_train)
    test = TensorDataset(input_tensor_test, target_tensor_test)

    # data loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, device)

    if torch.cuda.is_available():
        model.cuda()

    # Cross Entropy Loss
    error = nn.CrossEntropyLoss()

    # SGD Optimizer
    learning_rate = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_list = []
    iteration_list = []
    accuracy_list = []
    count = 0
    print("training start")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):


            train = Variable(images.view(-1, seq_len, input_dim)).requires_grad_()
            labels = Variable(labels)

            if torch.cuda.is_available():
                train.cuda()
                labels.cuda()

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

            if count % 50 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                valid_output_list = []
                for images, labels in test_loader:
                    images = Variable(images.view(-1, seq_len, input_dim))

                    # Forward propagation
                    outputs = model(images)

                    valid_output_list += outputs

                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]

                    # Total number of labels
                    total += labels.size(0)

                    correct += (predicted == labels.to(device=device)).sum()

                accuracy = 100 * correct / float(total)
                hit_rate = calc_hit_rate(valid_output_list, valid_ext_list_hit)

                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
                print('Iteration: {}  Loss: {}  Accuracy: {} % Hit_rate: {} %'.format(count, loss.data, accuracy, hit_rate))
                if count % 500 == 0:
                    torch.save(model.state_dict(), './model/seq_len/model_' + str(count) + '.pth')

def get_sub_list(output_list, metadata):
    sub = []
    sent_count = 0

    for _, len_article in metadata:
        result = []
        for sent_num in range(len_article):
            ext_pos = torch.nn.functional.softmax(output_list[sent_count], dim=0)[1].item()
            if len(result) >= 3:
                result = sorted(result, key=(lambda x: x[1]), reverse=True)
                if ext_pos > result[-1][1]:
                    result = result[:-1]
                    result.append((sent_num, ext_pos))
            else:
                result.append((sent_num, ext_pos))
            sent_count += 1
        if len(result) != 3:
            print(result)
        result = sorted(result, key=(lambda x: x[1]), reverse=True)
        sub.append(result)
    return sub


def calc_hit_rate(output_list, metadata):
    count_mom = 0
    count_son = 0
    sub = get_sub_list(output_list, metadata)
    print(len(sub), len(metadata))
    for n, result in enumerate(sub):
        for sub_num, _ in result:
            if sub_num in metadata[n][0]:
                count_son += 1
            count_mom += 1
    return count_son / count_mom


if __name__ == '__main__':
    train()