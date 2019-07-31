from models.nvidia_lstm import LSTMModel
from data_loading.F1Dataset import F1Dataset, ToTensor, Crop, Normalize
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn.modules.loss import MSELoss, SmoothL1Loss
import numpy as np
import matplotlib.pyplot as plt
import cv2


def normalize_weights(weights):
    return ((weights.max() + 1) - weights) / (weights.sum() + weights.size)


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)


base = 'C:\\Users\\Andrei\\workspace\\Robotics\\data\\F1_Australia_Line\\'

dataset = F1Dataset(base + 'driving_log.csv', base, seq_len=1, frame_delta=12, future_frame=5,
                    transform=transforms.Compose([Crop((0.4, 0.39, 0.13, 0.13)), ToTensor(), Normalize()]))

no_bins = 100

bins = np.linspace(0, 1, no_bins)

weights_acc = np.zeros(no_bins + 1)
weights_brake = np.zeros(no_bins + 1)
weights_steer = np.zeros(no_bins + 1)

for i, el in enumerate(dataset):
    acc, brake, steer = el['commands'][0]
    acc_bin = np.digitize(acc, bins)
    brake_bin = np.digitize(brake, bins)
    steer_bin = np.digitize(steer, bins)
    weights_acc[acc_bin] += 1
    weights_brake[brake_bin] += 1
    weights_steer[steer_bin] += 1

weights_acc = normalize_weights(weights_acc)
weights_brake = normalize_weights(weights_brake)
weights_steer = normalize_weights(weights_steer)

plt.plot(weights_acc, label='Acceleration weights')
plt.plot(weights_brake, label='Braking weights')
plt.plot(weights_steer, label='Steer weights')
plt.xlabel('Bin number')
plt.ylabel('Weight')
plt.legend()
plt.savefig('weights.png')
plt.show()


def find_weights(input, bins):
    input = input.cpu()
    input_indices = np.digitize(input, bins)
    weights_a = weights_acc[input_indices[:, 0]]
    weights_b = weights_brake[input_indices[:, 1]]
    weights_s = weights_steer[input_indices[:, 2]]
    return weights_a, weights_b, weights_s

dataset = F1Dataset(base + 'driving_log.csv', base, seq_len=3, frame_delta=10, future_frame=2,
                    transform=transforms.Compose([Crop((0.4, 0.39, 0.13, 0.13)), ToTensor(), Normalize()]))


shuffle_dataset = True
validation_split = 0.0

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=256, num_workers=0, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=256, num_workers=0, sampler=valid_sampler)

model = LSTMModel(no_outputs=1)
model = model.cuda()
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, 1, 0.999)
criterion = MSELoss()
criterion = criterion.cuda()

print_each = 20


def train(epoch):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        inputs = data['images'].cuda()
        targets = data['commands'].cuda()
        optimizer.zero_grad()

        weights_a, weights_b, weights_s = find_weights(targets[:, 0], bins)

        weights_a = torch.tensor(weights_a).cuda().float()
        weights_b = torch.tensor(weights_b).cuda().float()
        weights_s = torch.tensor(weights_s).cuda().float()

        outputs = model(inputs)
        loss = weighted_mse_loss(targets[:, :, 0], outputs[0], weights_a) + \
            weighted_mse_loss(targets[:, :, 1], outputs[1], weights_b) + \
            weighted_mse_loss(targets[:, :, 2], outputs[2], weights_s)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % print_each == print_each - 1:
            print('[%d, %5d] loss: %.6f' % (epoch, i + 1, running_loss / print_each))
            running_loss = 0.0
    torch.save(model.state_dict(), 'C:\\Users\\Andrei\\workspace\\Robotics\\checkpoints\\checkpoint_train_s_lstm_aug_%d' % (epoch))


global best_eval_loss
best_eval_loss = 10e7


def eval(epoch):
    global best_eval_loss
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            print(i)
            inputs = data['images'].cuda()
            targets = data['commands'].cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
        print('[%d] loss: %.6f' % (epoch, total_loss / (i + 1)))

        if total_loss < best_eval_loss:
            best_eval_loss = total_loss
            torch.save(model.state_dict(), 'C:\\Users\\Andrei\\workspace\\Robotics\\checkpoints\\checkpoint_s4%d' % (epoch))


if __name__ == '__main__':
    for epoch in range(500):
        train(epoch + 1)
        #scheduler.step()
        #eval(epoch + 1)