import pdb
import argparse
import numpy as np
from model import UNet
from datasets import data
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Calculate dice loss from one hot encoded predictions and integer labels
def dice_loss(pred, target):
    smooth = 1e-6
    pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def get_args():
    parser = argparse.ArgumentParser('Vessel Segmentation')
    parser.add_argument('train_data_dir', type=str, help='Directory for the train data')
    parser.add_argument('test_data_dir', type=str, help='Directory for the test data')
    parser.add_argument('--train-videos', default=None, nargs='+', type=str,
                        help='Videos to use for train. If empty all are used.')
    parser.add_argument('--test-videos', default=None, nargs='+', type=str,
                        help='Videos to use for test. If empty all are used.')
    parser.add_argument('--output-dir', default='output', type=str,
                        help='Directory to save the output masks')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='Learning rate for training')
    parser.add_argument('--num-epochs', default=20, type=int,
                        help='Number of epochs for training')
    parser.add_argument('--load-model', action='store_true',
                        help='Load model from file rather than training')
    parser.add_argument('--save-masks', action='store_true',
                        help='Save the output masks')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    return parser.parse_args()


def train(model, train_dataloader, val_dataloader, device,
          learning_rate=0.001, num_epochs=20, verbose=False):

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                           patience=5, verbose=True)

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
            if i % 10 == 0 and i > 0:
                last_loss = running_loss / 10
                if verbose:
                    print(f'\nbatch {i} running loss: {last_loss}', flush=True)
                running_loss = 0.
            else:
                print('.', end="", flush=True)

        return last_loss

    for epoch in range(num_epochs):

        model.train(True)
        avg_loss = train_one_epoch(epoch)

        running_vloss = 0.0
        running_dice = 0.0

        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)

                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                dice_score = dice_loss(voutputs, vlabels)
                running_vloss += vloss
                running_dice += dice_score

        avg_vloss = running_vloss / (i + 1)
        avg_dice = running_dice / (i + 1)
        print(f'Epoch {epoch} complete, train loss: {avg_loss} val_loss: {avg_vloss} '
                  f'val_dice: {avg_dice}', flush=True)
        scheduler.step(avg_vloss)


if __name__ == '__main__':

    args = get_args()

    model = UNet(3, 8)
    #model = UNet(5, 64)

    print(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device} for training.')

    model.to(device)

    train_dataset = data['vessel_data'](args.train_data_dir, videos=args.train_videos)
    test_dataset = data['vessel_data'](args.test_data_dir, videos=args.test_videos)

    train_set, val_set = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=2)
    #train_dataloader.to(device)
    #val_dataloader.to(device)

    if args.load_model:
        model.load_state_dict(torch.load(str(output_dir / 'unet_model.pth')))
        print('Model loaded from file.')
    else:
        train(model, train_dataloader, val_dataloader, device,
              args.learning_rate, args.num_epochs, args.verbose)

        # Save the model
        torch.save(model.state_dict(), str(output_dir / 'unet_model.pth'))
        print('Model trained and saved to file.')

    print('Testing the model...')
    # Test the model
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=2)

    n = 0
    sum_dice = 0
    for i, data in enumerate(test_dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        outputs = model(inputs).cpu()

        dice_score = dice_loss(outputs, labels)
        sum_dice += dice_score.item()

        if args.save_masks:
            # Save the produced segmentation masks
            for out, label in zip(outputs, labels):
                output_masks = out.cpu().detach() * 255
                label_mask = label * 50
                # Save the output mask and original labels
                for m, mask in enumerate(output_masks):
                    im = Image.fromarray(mask.numpy())
                    im = im.convert('L')
                    im.save(str(output_dir / 'masks' / f'output_mask_{n}_{m}.png'))
                Image.fromarray(label_mask.numpy().astype(np.uint8)).save(
                    str(output_dir / 'labels' / f'original_label_{n}.png'))
                n += 1
            print(f'Image {i} done, dice score: {dice_score.item()}')

    print(f'Average Dice score: {sum_dice / len(test_dataloader)}')
