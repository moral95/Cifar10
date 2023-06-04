import os
import torch
import torch.nn as nn
import torchvision
import random
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


# def getdata(batch_size, dataset, root_dir, resize, data_aug, cutout, autoaug):
def getdata(batch_size, root_dir, resize):    
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    seed_number = 7890
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define the data transforms
    transform_train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
#     if data_aug:
#         transform_list = [transforms.RandomCrop(32, padding=4, fill=128)]
#         transform_list.append(transforms.RandomHorizontalFlip())
#         if autoaug:
#             transform_list.append(CIFAR10Policy())
#         transform_list.append(transforms.ToTensor())
#         if cutout or autoaug:
#             transform_list.append(Cutout(n_holes=1, length=16))
#         transform_list.append(transforms.Normalize(mean, std))
#         transform_train = transforms.Compose(transform_list)
#         transform_test = transforms.Compose([
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean, std)
#                     ])
#     else:
#         transform_train = transforms.Compose([
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean, std)
#                     ])
#         transform_test = transforms.Compose([
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean, std)
#                     ])
        

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform_train)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform_test)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)
    
    return train_dataloader, test_dataloader

# class Cutout:

#     def __init__(self, n_holes, length):
#         self.n_holes = n_holes
#         self.length = length

#     def __call__(self, img):
#         ''' img: Tensor image of size (C, H, W) '''
#         _, h, w = img.size()
#         mask = np.ones((h, w), np.float32)
#         for n in range(self.n_holes):
#             y = np.random.randint(h)
#             x = np.random.randint(w)
#             y1 = int(np.clip(y - self.length // 2, 0, h))
#             y2 = int(np.clip(y + self.length // 2, 0, h))
#             x1 = int(np.clip(x - self.length // 2, 0, w))
#             x2 = int(np.clip(x + self.length // 2, 0, w))
#             mask[y1: y2, x1: x2] = 0.
#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img = img * mask
#         return img

# class CIFAR10Policy:
#     ''' Randomly choose one of the best 25 Sub-policies on CIFAR10. '''
#     def __init__(self, fillcolor=(128, 128, 128)):
#         self.policies = [
#             SubPolicy(0.1, 'invert', 7, 0.2, 'contrast', 6, fillcolor),
#             SubPolicy(0.7, 'rotate', 2, 0.3, 'translateX', 9, fillcolor),
#             SubPolicy(0.8, 'sharpness', 1, 0.9, 'sharpness', 3, fillcolor),
#             SubPolicy(0.5, 'shearY', 8, 0.7, 'translateY', 9, fillcolor),
#             SubPolicy(0.5, 'autocontrast', 8, 0.9, 'equalize', 2, fillcolor),

#             SubPolicy(0.2, 'shearY', 7, 0.3, 'posterize', 7, fillcolor),
#             SubPolicy(0.4, 'color', 3, 0.6, 'brightness', 7, fillcolor),
#             SubPolicy(0.3, 'sharpness', 9, 0.7, 'brightness', 9, fillcolor),
#             SubPolicy(0.6, 'equalize', 5, 0.5, 'equalize', 1, fillcolor),
#             SubPolicy(0.6, 'contrast', 7, 0.6, 'sharpness', 5, fillcolor),

#             SubPolicy(0.7, 'color', 7, 0.5, 'translateX', 8, fillcolor),
#             SubPolicy(0.3, 'equalize', 7, 0.4, 'autocontrast', 8, fillcolor),
#             SubPolicy(0.4, 'translateY', 3, 0.2, 'sharpness', 6, fillcolor),
#             SubPolicy(0.9, 'brightness', 6, 0.2, 'color', 8, fillcolor),
#             SubPolicy(0.5, 'solarize', 2, 0.0, 'invert', 3, fillcolor),

#             SubPolicy(0.2, 'equalize', 0, 0.6, 'autocontrast', 0, fillcolor),
#             SubPolicy(0.2, 'equalize', 8, 0.6, 'equalize', 4, fillcolor),
#             SubPolicy(0.9, 'color', 9, 0.6, 'equalize', 6, fillcolor),
#             SubPolicy(0.8, 'autocontrast', 4, 0.2, 'solarize', 8, fillcolor),
#             SubPolicy(0.1, 'brightness', 3, 0.7, 'color', 0, fillcolor),

#             SubPolicy(0.4, 'solarize', 5, 0.9, 'autocontrast', 3, fillcolor),
#             SubPolicy(0.9, 'translateY', 9, 0.7, 'translateY', 9, fillcolor),
#             SubPolicy(0.9, 'autocontrast', 2, 0.8, 'solarize', 3, fillcolor),
#             SubPolicy(0.8, 'equalize', 8, 0.1, 'invert', 3, fillcolor),
#             SubPolicy(0.7, 'translateY', 9, 0.9, 'autocontrast', 1, fillcolor)
#         ]

#     def __call__(self, img):
#         policy_idx = random.randint(0, len(self.policies) - 1)
#         return self.policies[policy_idx](img)

#     def __repr__(self):
#         return 'AutoAugment CIFAR10 Policy'


# class SubPolicy:

#     def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
#         ranges = {
#             'shearX': np.linspace(0, 0.3, 10),
#             'shearY': np.linspace(0, 0.3, 10),
#             'translateX': np.linspace(0, 150 / 331, 10),
#             'translateY': np.linspace(0, 150 / 331, 10),
#             'rotate': np.linspace(0, 30, 10),
#             'color': np.linspace(0.0, 0.9, 10),
#             'posterize': np.round(np.linspace(8, 4, 10), 0).astype(np.int),
#             'solarize': np.linspace(256, 0, 10),
#             'contrast': np.linspace(0.0, 0.9, 10),
#             'sharpness': np.linspace(0.0, 0.9, 10),
#             'brightness': np.linspace(0.0, 0.9, 10),
#             'autocontrast': [0] * 10,
#             'equalize': [0] * 10,
#             'invert': [0] * 10
#         }
#         def rotate_with_fill(img, magnitude):
#             rot = img.convert('RGBA').rotate(magnitude)
#             return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4), rot).convert(img.mode)

#         func = {
#             'shearX': lambda img, magnitude: img.transform(
#                 img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
#                 Image.BICUBIC, fillcolor=fillcolor),
#             'shearY': lambda img, magnitude: img.transform(
#                 img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
#                 Image.BICUBIC, fillcolor=fillcolor),
#             'translateX': lambda img, magnitude: img.transform(
#                 img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
#                 fillcolor=fillcolor),
#             'translateY': lambda img, magnitude: img.transform(
#                 img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
#                 fillcolor=fillcolor),
#             'rotate': lambda img, magnitude: rotate_with_fill(img, magnitude),
#             'color': lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
#             'posterize': lambda img, magnitude: ImageOps.posterize(img, magnitude),
#             'solarize': lambda img, magnitude: ImageOps.solarize(img, magnitude),
#             'contrast': lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
#                 1 + magnitude * random.choice([-1, 1])),
#             'sharpness': lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
#                 1 + magnitude * random.choice([-1, 1])),
#             'brightness': lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
#                 1 + magnitude * random.choice([-1, 1])),
#             'autocontrast': lambda img, magnitude: ImageOps.autocontrast(img),
#             'equalize': lambda img, magnitude: ImageOps.equalize(img),
#             'invert': lambda img, magnitude: ImageOps.invert(img)
#         }

#         self.p1 = p1
#         self.operation1 = func[operation1]
#         self.magnitude1 = ranges[operation1][magnitude_idx1]
#         self.p2 = p2
#         self.operation2 = func[operation2]
#         self.magnitude2 = ranges[operation2][magnitude_idx2]


#     def __call__(self, img):
#         if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
#         if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
#         return img
def get_result_save_dir(args):

    log_dir = f"/NasData/home/lsh/1.STUDY/1.1.Deep_Learning_Fundamentals/1.1.1.1.Cifar10_baseline/experimnet_results/{args.ckpt+args.model_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(log_dir+'/model')
    return log_dir

# def mini_batch_training(train_dataloader, test_dataloader, args, device):

#     #[save_Dir]
#     log_dir = get_result_save_dir(args)
#     # 로그 데이터는 앞의 preactresnet 참고하여 변경하자.

#         # Define the loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#     optimizer = torch.optim.Adam(params=model.parameters(), lr= lr, weight_decay=weight_decay)

#     # Define the learning rate scheduler
#     # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor= 0.1, patience= 10)

#     #[Save Dir]
#     log_dir = get_result_save_dir(args)

#     #[Optimizer]
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam([{'params': params_1x}, {'params':classifier_param, 'lr':args.lr*10}], lr=args.lr, weight_decay = args.weight_decay)
#     start = time.time()


#     results = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[]}
#     for e in range(1, args.epochs+1):
#         log = open(f'{log_dir}/log.txt', "a")
#         print(f'============={e}/{args.epochs}=============');log.write(f'============={e}/{args.epochs}=============\n')
#         #[Train]
#         model.train()
#         train_loss = torch.tensor(0., device = device)
#         train_accuracy = torch.tensor(0., device = device)
#         with tqdm(total = len(train_dataloader), desc ="training") as pbar:
#             for inputs, labels in train_dataloader:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 preds = model(inputs)
#                 loss = criterion(preds, labels)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 with torch.no_grad():
#                     train_loss += loss*train_dataloader.batch_size
#                     train_accuracy += (torch.argmax(preds, dim = 1) == labels).sum()
#                 pbar.update(1)

#         #[Test]
#         model.eval()
#         test_loss = torch.tensor(0., device = device)
#         test_accuracy = torch.tensor(0., device = device)

#         with torch.no_grad():
#             with tqdm(total=len(test_dataloader), desc = "Test") as pbar:
#                 for inputs, labels in test_dataloader:
#                     inputs = inputs.to(device)
#                     labels = labels.to(device)

#                     preds = model(inputs)
#                     loss = criterion(preds, labels)

#                     test_loss += loss*test_dataloader.batch_size
#                     test_accuracy += (torch.argmax(preds, dim = 1) == labels).sum()
#                     pbar.update(1)


#         train_loss = train_loss / len(train_dataloader.dataset)
#         train_accuracy = (train_accuracy/len(train_dataloader.dataset))*100 
#         print(f'[Training] loss: {train_loss:.2f}, accuracy: {train_accuracy:.3f}%'); log.write(f'[Training] loss: {train_loss:.2f}, accuracy: {train_accuracy:.3f}%\n')

#         test_loss = test_loss / len(test_dataloader.dataset)
#         test_accuracy = (test_accuracy/len(test_dataloader.dataset))*100 
#         print(f'[Test] loss: {test_loss:.2f}, accuracy: {test_accuracy:.3f} %'); log.write(f'[Test] loss: {test_loss:.2f}, accuracy: {test_accuracy:.3f} %\n')


#         #[Epoch Result Save]
#         train_loss, train_accuracy, test_loss, test_accuracy = train_loss.item(), train_accuracy.item(), test_loss.item(), test_accuracy.item()
#         train_loss, train_accuracy, test_loss, test_accuracy = round(train_loss ,3), round(train_accuracy,3), round(test_loss,3), round(test_accuracy,3)
#         results['train_loss'].append(train_loss); results['train_acc'].append(train_accuracy); results['test_loss'].append(test_loss); results['test_acc'].append(test_accuracy)
#         torch.save(model.state_dict(), log_dir+f'/model/{e}.pth')
#         log.close()

