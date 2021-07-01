import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import RNNModel
from LSTM import LSTMModel
from TransformerEncoder import TransformerEncoder
from rnn_tokenize import tokenize, create_dataset
import numpy as np
import random
import pickle

def train():
    # LSTM configs
    batch_size = 50
    n_iters = 20000
    visible_gpus = 0
    seed = 777
    input_dim = 10  # input dimension
    hidden_dim = 10  # hidden layer dimension
    layer_dim = 1  # number of hidden layers
    output_dim = 2  # output dimension
    seq_len = 5

    # RNN configs
    # batch_size = 16
    # n_iters = 40000
    # visible_gpus = 0
    # seed = 777
    # # Create RNN
    # input_dim = 10  # input dimension
    # hidden_dim = 10  # hidden layer dimension
    # layer_dim = 1  # number of hidden layers
    # output_dim = 2  # output dimension
    # seq_len = 4

    # batch_size = 256
    # n_iters = 20000
    # visible_gpus = 1
    # seed = 777
    # # Create RNN
    # input_dim = 10  # input dimension
    # hidden_dim = 10  # hidden layer dimension
    # layer_dim = 1  # number of hidden layers
    # output_dim = 2  # output dimension
    # seq_len = 10

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
    input_train, input_test, target_train, target_test, pad_mask_train, pad_mask_valid = create_dataset(
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
    # input_tensor_train = torch.Tensor(input_train).float()

    input_tensor_train = torch.from_numpy(np.array(input_train, dtype=np.float64)).float()
    input_tensor_test = torch.from_numpy(np.array(input_test, dtype=np.float64)).float()

    # target_tensor_train = torch.LongTensor(target_train)

    print("success")
    target_tensor_train = torch.from_numpy(np.array(target_train, dtype=np.float64)).float().type(torch.LongTensor)
    target_tensor_test = torch.from_numpy(np.array(target_test, dtype=np.float64)).float().type(torch.LongTensor)


    print(input_tensor_train.shape, target_tensor_train.shape)

    # print(input_tensor_train.shape, target_tensor_train.shape)

    pad_mask_train = torch.LongTensor(pad_mask_train)
    pad_mask_valid = torch.LongTensor(pad_mask_valid)

    # Pytorch train and test sets
    train = TensorDataset(input_tensor_train, target_tensor_train, pad_mask_train)
    test = TensorDataset(input_tensor_test, target_tensor_test, pad_mask_valid)

    print("TensorDataset created")


    # data loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    print("Dataloader created")

    # model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, device)
    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, seq_len, device)
    # model = TransformerEncoder(
    #     d_model=input_dim,
    #     d_ff=hidden_dim,
    #     heads=1,
    #     dropout=0.1,
    #     num_inter_layers=layer_dim,
    #     output_dim=output_dim,
    #     device=device
    # )

    linear_layer = nn.Linear(hidden_dim, output_dim)

    if torch.cuda.is_available():
        model.to(device=f"cuda:{visible_gpus}")
        linear_layer.to(device=f"cuda:{visible_gpus}")

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
        for i, (images, labels, pad_train) in enumerate(train_loader):

            # print("input image", images.shape)
            # print("input labels", labels.shape)
            # print("input pad", pad_train.shape)
            train = Variable(images.view(-1, seq_len, input_dim)).requires_grad_()
            # train = Variable(images).requires_grad_()
            labels = Variable(labels)
            # print("var image", train.shape)
            # print("var labels", labels.shape)
            # print("input pad", pad_train.shape)

            if torch.cuda.is_available():
                train.to(device=f"cuda:{visible_gpus}")
                labels.to(device=f"cuda:{visible_gpus}")

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(train)

            # print(outputs.shape)
            # print(labels.shape)

            # labels = torch.flatten(labels)
            # if batch_size * (i+1) <= len(pad_mask_train):
            #     pad_mask_train = pad_mask_train[batch_size * i:batch_size * (i+1)]
            # else:
            #     print("else")
            #     pad_mask_train = pad_mask_train[batch_size * i:len(pad_mask_train)]
            # print("pad", pad_train.shape)
            # print("label", labels.shape)
            # print("output", outputs.shape)
            # nonzeros = torch.nonzero(torch.LongTensor(pad_train), as_tuple=True)
            # labels = labels[nonzeros[0], nonzeros[1]]
            # # print("label gather", labels.shape)
            # # labels = torch.flatten(labels)
            # # print("label flatten", labels.shape)
            # outputs = outputs[nonzeros[0], nonzeros[1]]

            # outputs = linear_layer(outputs)
            # outputs = outputs[nonzeros[0]]


            # print(labels.shape)

            # Calculate softmax and ross entropy loss

            # print("output", outputs)
            # print("label", labels)

            loss = error(outputs, labels.to(device=device)) # .to(device=device)

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
                for i, (images, labels, pad_valid) in enumerate(test_loader):
                    # print("labels", labels)
                    images = Variable(images.view(-1, seq_len, input_dim))
                    # images = Variable(images).requires_grad_()

                    # Forward propagation
                    outputs = model(images)

                    # print("output", outputs.shape)

                    # outputs = outputs[nonzeros[0], nonzeros[1]]
                    # nonzeros = torch.nonzero(torch.LongTensor(pad_valid), as_tuple=True)
                    # # outputs = outputs[nonzeros[0]]
                    # outputs = outputs[nonzeros[0], nonzeros[1]]
                    # outputs = linear_layer(outputs)

                    valid_output_list += outputs

                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]

                    # print("test")
                    #
                    # print("label", labels.shape)
                    # labels = torch.flatten(labels)
                    # labels = labels[nonzeros[0], nonzeros[1]]
                    # print("labels size", labels.shape)
                    # labels = torch.flatten(labels)
                    # print("labels flatten size", labels.shape)

                    # print("flatten", labels.shape)
                    # print("size 0", labels.size(0))
                    # Total number of labels
                    # print("label", labels.shape)
                    total += labels.size(0)

                    # print("predicted size", predicted.shape)

                    correct += (predicted == labels.to(device=device)).sum()

                # print("end")

                if total == 0:
                    print("Can't calculate accuracy & hit rate")
                else:
                    accuracy = 100 * correct / float(total)
                    # print("output", valid_output_list)
                    # print("hist", valid_ext_list_hit)
                    hit_rate = calc_hit_rate(valid_output_list, valid_ext_list_hit)

                # store loss and iteration
                    loss_list.append(loss.data)
                    iteration_list.append(count)
                    accuracy_list.append(accuracy)
                    print('Iteration: {}  Loss: {}  Accuracy: {} % Hit_rate: {} %'.format(count, loss.data, accuracy, hit_rate))
                if count % 500 == 0:
                    torch.save(model.state_dict(), './model/article_rnn/model_' + str(count) + '.pth')

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