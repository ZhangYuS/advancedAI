from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import  models, transforms
from sklearn.metrics import f1_score

import time
import os

from PIL import Image
import random
from efficientnet_pytorch import EfficientNet

from Focal_Loss import focal_loss

# use PIL Image to read image

def calculate_F1(pred_label, golden_label, label_num):
    TP = torch.sum((pred_label==label_num)&(golden_label==label_num))
    TN = torch.sum((pred_label!=label_num)&(golden_label!=label_num))
    FN = torch.sum((pred_label!=label_num)&(golden_label==label_num))
    FP = torch.sum((pred_label==label_num)&(golden_label!=label_num))
    pre = 0 if (TP.item() + FP.item()) == 0 else TP.item() / (TP.item() + FP.item())
    rec = 0 if (TP.item() + FN.item()) == 0 else TP.item() / (TP.item() + FN.item())
    F1 = 0 if (pre + rec) == 0 else (2 * pre * rec) / (pre + rec)
    return {'pre': pre, 'rec': rec, 'F1': F1}

def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader, label_index=None):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            self.img_label = [int(line.strip().split('\t')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader
        self.label_index = label_index

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label

    def undersampling(self):
        data_pair = list(zip(self.img_name, self.img_label))
        label_data = {}
        for idx, label in enumerate(self.label_index):
            label_data[label] = list(filter(lambda x: x[1]==idx, data_pair))
        label_data_num = {x[0]: len(x[1]) for x in label_data.items()}
        balance_num = min(label_data_num.values())
        balance_data_pair = []
        for label in self.label_index:
            if label_data_num[label] > balance_num:
                label_data[label] = random.choices(label_data[label], k=balance_num)
            else:
                label_data[label] = (label_data[label] * (balance_num // label_data_num[label]))[: balance_num]
            balance_data_pair += label_data[label]
        self.img_name, self.img_label = zip(*balance_data_pair)


    def oversampling(self):
        data_pair = list(zip(self.img_name, self.img_label))
        label_data = {}
        for idx, label in enumerate(self.label_index):
            label_data[label] = list(filter(lambda x: x[1] == idx, data_pair))
        label_data_num = {x[0]: len(x[1]) for x in label_data.items()}
        balance_num = max(label_data_num.values())
        balance_data_pair = []
        for label in self.label_index:
            if label_data_num[label] > balance_num:
                label_data[label] = random.choices(label_data[label], k=balance_num)
            else:
                label_data[label] = (label_data[label] * (balance_num // label_data_num[label]))[: balance_num]
            balance_data_pair += label_data[label]
        self.img_name, self.img_label = zip(*balance_data_pair)


def train_model(model, criterion, optimizer, scheduler, num_epochs, use_gpu, label_index=None):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_f1 = 0.0
    best_cat_f1 = 0.0

    for epoch in range(num_epochs):
        begin_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        preds_all = None
        labels_all = None
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            count_batch = 0

            # Iterate over data.
            for data in dataloders[phase]:
                count_batch += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

                if preds_all is None:
                    preds_all = preds.cpu().detach()
                    labels_all = labels.data.cpu().detach()
                else:
                    preds_all = torch.cat((preds_all, preds.cpu().detach()), dim=0)
                    labels_all = torch.cat((labels_all, labels.data.cpu().detach()), dim=0)

                # print result every 10 batch
                if count_batch%10 == 0:
                    batch_loss = running_loss / (batch_size*count_batch)
                    batch_acc = running_corrects / (batch_size*count_batch)
                    print('{} Epoch [{}] Batch [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'. \
                          format(phase, epoch, count_batch, batch_loss, batch_acc, time.time()-begin_time))
                    begin_time = time.time()
                    # for idx, label in enumerate(label_index):
                    #     F1 = calculate_F1(preds, labels.data, idx)
                    #     print('{} Epoch [{}] Batch [{}] {} pre: {:.4f} rec: {:.4f} F1: {:.4f}s'. \
                    #           format(phase, epoch, count_batch, label, F1['pre'], F1['rec'], F1['F1']))
                    # print('-' * 30)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_f1_all = f1_score(labels_all, preds_all, labels= [0, 1, 2],average=None)
            epoch_f1_macro = f1_score(labels_all, preds_all, average='macro')

            for l, f1 in zip(['human', 'cat', 'dog'], epoch_f1_all):
                print(f'{l}: {f1}')
            print(f'f1_all: {epoch_f1_all}')

            print(f'F1: {epoch_f1_macro}')
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # save model
            if phase == 'train':
                if not os.path.exists('output'):
                    os.makedirs('output')
                torch.save(model, 'output/resnet_epoch{}.pkl'.format(epoch))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

            if phase == 'val' and epoch_f1_macro > best_f1:
                best_f1 = epoch_f1_macro
                #best_model_wts = model.state_dict()

            if phase == 'val' and epoch_f1_all[label_index.index('cat')] > best_cat_f1:
                best_cat_f1 = epoch_f1_all[label_index.index('cat')]


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val F1: {:4f}'.format(best_f1))
    print('Best cat F1: {:4f}'.format(best_cat_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    seed = 2000
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    use_gpu = torch.cuda.is_available()

    batch_size = 64
    num_class = 3
    label_index = ['human', 'cat', 'dog']
    # data_file = 'data'
    data_file = 'unbalanced_data/06-17-18-03-07_50_300_300'
    # data_file = 'unbalanced_data/06-19-15-57-09_300_50_300'
    # data_file = 'unbalanced_data/06-19-16-00-51_300_300_50'

    image_datasets = {x: customData(img_path=os.path.join(data_file, x, 'processed'),
                                    txt_path=(os.path.join(data_file, x, x + '_file_list.txt')),
                                    data_transforms=data_transforms,
                                    dataset=x, label_index=label_index) for x in ['train', 'val']}
    # image_datasets['train'].oversampling()
    # image_datasets['train'].undersampling()

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=True) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # get model and replace the original fc layer with your fc layer
    # model_ft = models.resnet50(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, num_class)

    efficient_net = EfficientNet.from_pretrained('efficientnet-b0')

    model_ft = nn.Sequential(
        efficient_net,
        nn.Linear(1000, 3)
    )

    # if use gpu
    if use_gpu:
        model_ft = model_ft.cuda()

    # define cost function
    # criterion = nn.CrossEntropyLoss()
    criterion = focal_loss(alpha=[1, 12, 1], gamma=2, num_classes=3)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.2 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.2)

    # multi-GPU
    # model_ft = torch.nn.DataParallel(model_ft, device_ids=[0,1])

    # train model
    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=25,
                           use_gpu=use_gpu,
                           label_index=label_index)

    # save best model
    torch.save(model_ft,"output/best_resnet.pkl")