from torchvision import datasets, transforms
import torch


def data_loader(train_dir, valid_dir, test_dir, batch):

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(p=0.18),
                                           transforms.RandomVerticalFlip(p=0.18),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

    data_transforms_valid = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    data_transforms_test = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)
    test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms_test)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloader_train = torch.utils.data.DataLoader(train_datasets, batch_size=batch, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(valid_datasets, batch_size=batch)
    dataloader_test = torch.utils.data.DataLoader(test_datasets, batch_size=batch)


    return dataloader_train, dataloader_valid, dataloader_test, train_datasets



