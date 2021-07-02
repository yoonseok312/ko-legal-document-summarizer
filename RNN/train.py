import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import RNNModel
from LSTM import LSTMModel
from rnn_tokenize import tokenize
#from rnn_tokenize import create_dataset
import numpy as np
import random
import pickle
from tensorboardX import SummaryWriter
from sentence_dataset import SentenceDataset
import datetime

now = datetime.datetime.now().strftime('%H-%M')
def train():
    #tensorboard settings
    writer =  SummaryWriter()

    # LSTM configs
    batch_size = 256
    n_iters = 50000
    visible_gpus = 0
    seed = 7777

    # Create RNN
    input_dim = 128  # input dimension
    hidden_dim = 256 # hidden layer dimension
    layer_dim = 6  # number of hidden layers
    output_dim = 2  # output dimension
    seq_len = 80

    # # RNN configs
    # batch_size = 32
    # n_iters = 40000
    # visible_gpus = 0
    # seed = 777
    # # Create RNN
    # input_dim = 128  # input dimension
    # hidden_dim = 256  # hidden layer dimension
    # layer_dim = 4  # number of hidden layers
    # output_dim = 2  # output dimension
    # seq_len = 40

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

    device = "cpu" if visible_gpus == '-1' else f"cuda"
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

    tokenized_data, tokenized_valid_data, embedding_model, target_train, target_test, _ = tokenize(input_dim)

    # print("opening  valid_ext_list_hit")
    # with open("./data/valid_ext_list_hit", "rb") as f:
    #     valid_ext_list_hit = pickle.load(f)

    num_epochs = n_iters / (len(tokenized_data) / batch_size)
    num_epochs = 100


    # Pytorch train and test sets
    print("-----Initializing SentenceDataset")
    train = SentenceDataset(tokenized_data, target_train, embedding_model, seq_len, input_dim)
    test = SentenceDataset(tokenized_valid_data, target_test, embedding_model, seq_len, input_dim)

    print("-----Initializing dataloader")
    # data loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=16)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=16)

    print("-----Initializing lstm model")
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    # model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, device)

    if torch.cuda.is_available():
        model = model.to(device=f"cuda")

    # Cross Entropy Loss
    error = nn.CrossEntropyLoss()

    # Adam Optimizer
    learning_rate = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    loss_list = []
    iteration_list = []
    accuracy_list = []
    count = 0
    print("-----Training start")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            train = images.view(-1, seq_len, input_dim)

            if torch.cuda.is_available():
                train, labels = train.to(device=f"cuda"), labels.to(device=f"cuda")

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            # Initialize hidden state with zeros
            #h0 = torch.zeros(layer_dim, train.size(0), hidden_dim, requires_grad=True).to(device="cuda")
            # Initialize cell state
            #c0 = torch.zeros(layer_dim, train.size(0), hidden_dim, requires_grad=True).to(device="cuda")

            h0 = torch.zeros(layer_dim, train.size(0), hidden_dim).to(device="cuda")
            c0 = torch.zeros(layer_dim, train.size(0), hidden_dim).to(device="cuda")
            outputs = model(train, h0, c0)

            # Calculate softmax and ross entropy loss

            loss = error(outputs, labels)

            # Calculating gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            count += 1
            
            writer.add_scalar('train loss', loss.data ,count)

            if count % 100 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                #valid_output_list = []
                with torch.no_grad():
                    model.eval()
                    for images_, labels_ in test_loader:
                        images_, labels_ = images_.to("cuda"), labels_.to("cuda")
                        total += labels_.size(0)
                        images_ = images_.view(-1, seq_len, input_dim)
                        h0_ = torch.zeros(layer_dim, images_.size(0), hidden_dim).to(device="cuda")
                        c0_ = torch.zeros(layer_dim, images_.size(0), hidden_dim).to(device="cuda")
                        outputs_ = model(images_, h0_, c0_)
                        # Get predictions from the maximum value
                        predicted = torch.max(outputs_.data, 1)[1]
                        # Total number of labels
                        #valid_output_list += outputs_
                        correct += torch.eq(predicted, labels_).sum().item()

                    accuracy = 100 * correct / float(total)
                    #hit_rate = calc_hit_rate(valid_output_list, valid_ext_list_hit)

                    writer.add_scalar('accuracy', accuracy, count)

                    # store loss and iteration
                    loss_list.append(loss.data)
                    iteration_list.append(count)
                    accuracy_list.append(accuracy)
                    print('Epoch: {} Iteration: {}  Loss: {}  Accuracy: {} % Hit_rate: {} %'.format(epoch, count, loss.data, accuracy, 100))
                    if count % 500 == 0:
                        torch.save(model.state_dict(), f'./model/seq_len_{seq_len}/model_{str(count)}_{now}.pth')


    writer.close()
def get_sub_list(output_list, metadata):
    sub = []
    sent_count = 0

    for _, len_article in metadata:
        result = []
        for sent_num in range(len_article):
            ext_pos = torch.nn.functional.sigmoid(output_list[sent_count])[1].item()
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
    #print(len(sub), len(metadata))
    for n, result in enumerate(sub):
        for sub_num, _ in result:
            if sub_num in metadata[n][0]:
                count_son += 1
            count_mom += 1
    return count_son / count_mom



if __name__ == '__main__':
    train()