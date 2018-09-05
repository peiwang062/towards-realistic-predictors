import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import os
from PIL import Image
import scipy.io as sio
import numpy as np


_NUM_CLASSES = {
    'imagenet': 1000,
    'indoor': 67,
}


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


def discarding_samples(train_list, val_list, hardness_scores_tr, hardness_scores_val, hardness_scores_idx_tr, hardness_scores_idx_val, percentage, dataset):

    # find threshold
    remaining_num_tr = int(hardness_scores_tr.shape[0] * percentage)
    hardness_scores_tr_tmp = hardness_scores_tr[np.argsort(hardness_scores_tr)]
    threshold = hardness_scores_tr_tmp[remaining_num_tr]

    # make realistic dataset list
    # realistic training set
    hardness_scores_tr = np.expand_dims(hardness_scores_tr, axis=0)
    hardness_scores_idx_tr = np.expand_dims(hardness_scores_idx_tr, axis=0)
    score_index = np.concatenate((hardness_scores_tr, hardness_scores_idx_tr), axis=0)
    score_index = score_index.transpose()
    score_index = score_index[np.argsort(score_index[:, 1])]
    scores = np.squeeze(score_index[:, 0])
    scores = scores.tolist()

    filenames = []
    labels = []
    with open(train_list, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            info = line.split()
            filenames.append(info[0])
            labels.append(int(info[1]))

    train_info = zip(filenames, labels, scores)
    train_info = sorted(train_info, key=lambda train: train[2])
    filenames, labels, scores = [list(l) for l in zip(*train_info)]

    filenames = filenames[0:remaining_num_tr]
    labels = labels[0:remaining_num_tr]

    realistic_train_list = './' + dataset + '/' + 'rea_img_label_' + str(int(100.0*percentage)) + '_tr.txt'
    fl = open(realistic_train_list, 'w')
    for i in range(len(labels)):
        save_info = filenames[i] + " " + str(labels[i])
        fl.write(save_info)
        fl.write("\n")
    fl.close()

    # realistic validation set
    hardness_scores_val = np.expand_dims(hardness_scores_val, axis=0)
    hardness_scores_idx_val = np.expand_dims(hardness_scores_idx_val, axis=0)
    score_index = np.concatenate((hardness_scores_val, hardness_scores_idx_val), axis=0)
    score_index = score_index.transpose()
    score_index = score_index[np.argsort(score_index[:, 1])]
    scores = np.squeeze(score_index[:, 0])
    scores = scores.tolist()

    filenames = []
    labels = []
    with open(val_list, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            info = line.split()
            filenames.append(info[0])
            labels.append(int(info[1]))

    val_info = zip(filenames, labels, scores)
    val_info = sorted(val_info, key=lambda val: val[2])
    filenames, labels, scores = [list(l) for l in zip(*val_info)]

    scores = np.array(scores)
    remaining_num_val = np.sum(scores < threshold)
    filenames = filenames[0:remaining_num_val]
    labels = labels[0:remaining_num_val]

    realistic_val_list = './' + dataset + '/' + 'rea_img_label_' + str(int(100.0*percentage)) + '_val.txt'
    fl = open(realistic_val_list, 'w')
    for i in range(len(labels)):
        save_info = filenames[i] + " " + str(labels[i])
        fl.write(save_info)
        fl.write("\n")
    fl.close()

    return realistic_train_list, realistic_val_list


def discarding_samples_cs(train_list, val_list, confidence_scores_tr, confidence_scores_val, confidence_scores_idx_tr, confidence_scores_idx_val, percentage, dataset):

    # find threshold
    remaining_num_tr = int(confidence_scores_tr.shape[0] * percentage)
    confidence_scores_tr_tmp = confidence_scores_tr[np.argsort(confidence_scores_tr)]
    threshold = confidence_scores_tr_tmp[-remaining_num_tr]

    # make realistic dataset list
    # realistic training set
    confidence_scores_tr = np.expand_dims(confidence_scores_tr, axis=0)
    confidence_scores_idx_tr = np.expand_dims(confidence_scores_idx_tr, axis=0)
    score_index = np.concatenate((confidence_scores_tr, confidence_scores_idx_tr), axis=0)
    score_index = score_index.transpose()
    score_index = score_index[np.argsort(score_index[:, 1])]
    scores = np.squeeze(score_index[:, 0])
    scores = scores.tolist()

    filenames = []
    labels = []
    with open(train_list, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            info = line.split()
            filenames.append(info[0])
            labels.append(int(info[1]))

    train_info = zip(filenames, labels, scores)
    train_info = sorted(train_info, key=lambda train: train[2])
    filenames, labels, scores = [list(l) for l in zip(*train_info)]

    filenames = filenames[-remaining_num_tr:]
    labels = labels[-remaining_num_tr:]

    realistic_train_list = './' + dataset + '/' + 'rea_img_label_' + str(int(100.0*percentage)) + '_standard_cnn_tr.txt'
    fl = open(realistic_train_list, 'w')
    for i in range(len(labels)):
        save_info = filenames[i] + " " + str(labels[i])
        fl.write(save_info)
        fl.write("\n")
    fl.close()

    # realistic validation set
    confidence_scores_val = np.expand_dims(confidence_scores_val, axis=0)
    confidence_scores_idx_val = np.expand_dims(confidence_scores_idx_val, axis=0)
    score_index = np.concatenate((confidence_scores_val, confidence_scores_idx_val), axis=0)
    score_index = score_index.transpose()
    score_index = score_index[np.argsort(score_index[:, 1])]
    scores = np.squeeze(score_index[:, 0])
    scores = scores.tolist()

    filenames = []
    labels = []
    with open(val_list, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            info = line.split()
            filenames.append(info[0])
            labels.append(int(info[1]))

    val_info = zip(filenames, labels, scores)
    val_info = sorted(val_info, key=lambda val: val[2])
    filenames, labels, scores = [list(l) for l in zip(*val_info)]

    scores = np.array(scores)
    remaining_num_val = np.sum(scores > threshold)
    filenames = filenames[-remaining_num_val:]
    labels = labels[-remaining_num_val:]

    realistic_val_list = './' + dataset + '/' + 'rea_img_label_' + str(int(100.0*percentage)) + '_standard_cnn_val.txt'
    fl = open(realistic_val_list, 'w')
    for i in range(len(labels)):
        save_info = filenames[i] + " " + str(labels[i])
        fl.write(save_info)
        fl.write("\n")
    fl.close()

    return realistic_train_list, realistic_val_list


class ImageFilelist(data.Dataset):
    def __init__(self, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


def imagenet(batch_size, train=True, val=True, percentage=0.9, thres_type=0, net_type=0, **kwargs):

    train_list = 'imagenet_gt_tr.txt'
    val_list = 'imagenet_gt_val.txt'

    if thres_type == 0:
        # using our hardness scores
        if net_type == 0:
            # resnet
            hardness_scores_tr = sio.loadmat('./imagenet/hardness_scores_res_res_tr2.mat')
            hardness_scores_val = sio.loadmat('./imagenet/hardness_scores_res_res_val2.mat')
            hardness_scores_idx_tr = sio.loadmat('./imagenet/hardness_scores_idx_res_res_tr2.mat')
            hardness_scores_idx_val = sio.loadmat('./imagenet/hardness_scores_idx_res_res_val2.mat')
        else:

            # vgg
            hardness_scores_tr = sio.loadmat('./imagenet/hardness_scores_vgg_vgg_tr.mat')
            hardness_scores_val = sio.loadmat('./imagenet/hardness_scores_vgg_vgg_val.mat')
            hardness_scores_idx_tr = sio.loadmat('./imagenet/hardness_scores_idx_vgg_vgg_tr.mat')
            hardness_scores_idx_val = sio.loadmat('./imagenet/hardness_scores_idx_vgg_vgg_val.mat')

        hardness_scores_tr = hardness_scores_tr['hardness_scores_tr']
        hardness_scores_tr = np.squeeze(hardness_scores_tr)
        hardness_scores_val = hardness_scores_val['hardness_scores_val']
        hardness_scores_val = np.squeeze(hardness_scores_val)
        hardness_scores_idx_tr = hardness_scores_idx_tr['hardness_scores_idx_tr']
        hardness_scores_idx_tr = np.squeeze(hardness_scores_idx_tr)
        hardness_scores_idx_val = hardness_scores_idx_val['hardness_scores_idx_val']
        hardness_scores_idx_val = np.squeeze(hardness_scores_idx_val)
        realistic_train_list, realistic_val_list = discarding_samples(train_list, val_list, hardness_scores_tr, hardness_scores_val, hardness_scores_idx_tr, hardness_scores_idx_val, percentage, 'imagenet')
    else:
        # using confidence scores
        if net_type == 0:
            # resnet
            confidence_scores_tr = sio.loadmat('./imagenet/confidence_scores_res_tr.mat')
            confidence_scores_val = sio.loadmat('./imagenet/confidence_scores_res_val.mat')
            confidence_scores_idx_tr = sio.loadmat('./imagenet/confidence_scores_idx_res_tr.mat')
            confidence_scores_idx_val = sio.loadmat('./imagenet/confidence_scores_idx_res_val.mat')
        else:
            # vgg
            confidence_scores_tr = sio.loadmat('./imagenet/confidence_scores_vgg_tr.mat')
            confidence_scores_val = sio.loadmat('./imagenet/confidence_scores_vgg_val.mat')
            confidence_scores_idx_tr = sio.loadmat('./imagenet/confidence_scores_idx_vgg_tr.mat')
            confidence_scores_idx_val = sio.loadmat('./imagenet/confidence_scores_idx_vgg_val.mat')

        confidence_scores_tr = confidence_scores_tr['confidence_scores_tr']
        confidence_scores_tr = np.squeeze(confidence_scores_tr)
        confidence_scores_val = confidence_scores_val['confidence_scores_val']
        confidence_scores_val = np.squeeze(confidence_scores_val)
        confidence_scores_idx_tr = confidence_scores_idx_tr['confidence_scores_index_tr']
        confidence_scores_idx_tr = np.squeeze(confidence_scores_idx_tr)
        confidence_scores_idx_val = confidence_scores_idx_val['confidence_scores_index_val']
        confidence_scores_idx_val = np.squeeze(confidence_scores_idx_val)
        realistic_train_list, realistic_val_list = discarding_samples_cs(train_list, val_list, confidence_scores_tr,
                                                                         confidence_scores_val, confidence_scores_idx_tr,
                                                                         confidence_scores_idx_val, percentage, 'imagenet')


    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=realistic_train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=realistic_val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def indoor(batch_size, train=True, val=True, percentage=0.9, thres_type=0, net_type=0, **kwargs):

    train_list = 'indoor_gt_tr.txt'
    val_list = 'indoor_gt_val.txt'

    if thres_type == 0:
        # using our hardness scores
        if net_type == 0:
            # resnet
            hardness_scores_tr = sio.loadmat('./indoor/hardness_scores_res_res_tr2.mat')
            hardness_scores_val = sio.loadmat('./indoor/hardness_scores_res_res_val2.mat')
            hardness_scores_idx_tr = sio.loadmat('./indoor/hardness_scores_idx_res_res_tr2.mat')
            hardness_scores_idx_val = sio.loadmat('./indoor/hardness_scores_idx_res_res_val2.mat')
        else:
            # vgg
            hardness_scores_tr = sio.loadmat('./indoor/hardness_scores_vgg_vgg_tr2.mat')
            hardness_scores_val = sio.loadmat('./indoor/hardness_scores_vgg_vgg_val2.mat')
            hardness_scores_idx_tr = sio.loadmat('./indoor/hardness_scores_idx_vgg_vgg_tr2.mat')
            hardness_scores_idx_val = sio.loadmat('./indoor/hardness_scores_idx_vgg_vgg_val2.mat')

        hardness_scores_tr = hardness_scores_tr['hardness_scores_tr']
        hardness_scores_tr = np.squeeze(hardness_scores_tr)
        hardness_scores_val = hardness_scores_val['hardness_scores_val']
        hardness_scores_val = np.squeeze(hardness_scores_val)
        hardness_scores_idx_tr = hardness_scores_idx_tr['hardness_scores_idx_tr']
        hardness_scores_idx_tr = np.squeeze(hardness_scores_idx_tr)
        hardness_scores_idx_val = hardness_scores_idx_val['hardness_scores_idx_val']
        hardness_scores_idx_val = np.squeeze(hardness_scores_idx_val)
        realistic_train_list, realistic_val_list = discarding_samples(train_list, val_list, hardness_scores_tr, hardness_scores_val, hardness_scores_idx_tr, hardness_scores_idx_val, percentage, 'indoor')
    else:
        # using confidence scores
        if net_type == 0:
            # resnet
            confidence_scores_tr = sio.loadmat('./indoor/confidence_scores_res_tr.mat')
            confidence_scores_val = sio.loadmat('./indoor/confidence_scores_res_val.mat')
            confidence_scores_idx_tr = sio.loadmat('./indoor/confidence_scores_idx_res_tr.mat')
            confidence_scores_idx_val = sio.loadmat('./indoor/confidence_scores_idx_res_val.mat')
        else:
            # vgg
            confidence_scores_tr = sio.loadmat('./indoor/confidence_scores_vgg_tr.mat')
            confidence_scores_val = sio.loadmat('./indoor/confidence_scores_vgg_val.mat')
            confidence_scores_idx_tr = sio.loadmat('./indoor/confidence_scores_idx_vgg_tr.mat')
            confidence_scores_idx_val = sio.loadmat('./indoor/confidence_scores_idx_vgg_val.mat')

        confidence_scores_tr = confidence_scores_tr['confidence_scores_tr']
        confidence_scores_tr = np.squeeze(confidence_scores_tr)
        confidence_scores_val = confidence_scores_val['confidence_scores_val']
        confidence_scores_val = np.squeeze(confidence_scores_val)
        confidence_scores_idx_tr = confidence_scores_idx_tr['confidence_scores_index_tr']
        confidence_scores_idx_tr = np.squeeze(confidence_scores_idx_tr)
        confidence_scores_idx_val = confidence_scores_idx_val['confidence_scores_index_val']
        confidence_scores_idx_val = np.squeeze(confidence_scores_idx_val)
        realistic_train_list, realistic_val_list = discarding_samples_cs(train_list, val_list, confidence_scores_tr,
                                                                         confidence_scores_val, confidence_scores_idx_tr,
                                                                         confidence_scores_idx_val, percentage, 'indoor')


    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=realistic_train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=realistic_val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds
