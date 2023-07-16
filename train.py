import argparse

import torch
from torch.utils.data import DataLoader

from model import UNet
from datasets import data


def get_args():
    parser = argparse.ArgumentParser('Vessel Segmentation')
    parser.add_argument('train_data_dir', type=str, help='Directory for the train data')
    parser.add_argument('test_data_dir', type=str, help='Directory for the test data')
    parser.add_argument('--train-videos', default=None, nargs='+', type=str,
                        help='Videos to use for train. If empty all are used.')
    parser.add_argument('--test-videos', default=None, nargs='+', type=str,
                        help='Videos to use for test. If empty all are used.')
    parser.add_argument('--batch-size', default=32, type=int)
    return parser.parse_args()

def train(model, train_dataloader, val_dataloader, device, learning_rate=0.001, num_epochs=20):

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_one_epoch(epoch_index):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(train_dataloader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                last_loss = running_loss / 100
                print(f'batch {i} loss: {last_loss}', flush=True)
                running_loss = 0.
            elif i % 10 == 9:
                print('.', end="", flush=True)

        return last_loss

    for epoch in range(num_epochs):

        model.train(True)
        avg_loss = train_one_epoch(epoch)

        running_vloss = 0.0

        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)

                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (epoch + 1)
        print(f'train loss: {avg_loss} val_loss: {avg_vloss}')


if __name__ == '__main__':

    args = get_args()

    model = UNet(3, 8)

    print(model)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device} for training.')

    model.to(device)

    train_dataset = data['vessel_data'](args.train_data_dir)
    #test_dataset = data['vessel_data'](args.test_data_dir)

    train_set, val_set = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=10)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=10)
    #train_dataloader.to(device)
    #val_dataloader.to(device)

    train(model, train_dataloader, val_dataloader, device)
