import importlib

import torch

from models.base_model import BaseModel


def get_dataloader(dataset_name,
                   dataset_train_root,
                   dataset_label_root,
                   transforms,
                   split='train',
                   batch_size=8,
                   num_workers=10,
                   train_on_train_id=False,
                   drop_last=True,
                   **kwargs
                   ):
    "Get a training dataloader"
    modul = importlib.import_module("datasets.expert_datasets")
    dataset_obj = getattr(modul, dataset_name)
    dataset = dataset_obj(img_root=dataset_train_root,
                          label_root=dataset_label_root,
                          split=split,
                          transform=transforms,
                          train_on_train_id=train_on_train_id,
                          **kwargs)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=True if split == 'train' else False,
                                         pin_memory=True,
                                         num_workers=num_workers,
                                         drop_last=drop_last)
    return loader


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".
    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    This function is taken from CycleGAN Repo
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/9bcef69d5b39385d18afad3d5a839a02ae0b43e7
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." %
              (model_filename, target_model_name))
        exit(0)

    return model


def get_model(config, device):
    "Loads module dependig von module name"
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model_obj = find_model_using_name(config.model_class)
    net = model_obj(config, config.input_channels)
    print(f"model {type(net).__name__} was created")

    if config.reset_weights:
        net.reset_weights()
    if config.uniform_init:
        net.init_weigths()

    net.to(device)

    return net


def get_loss(loss_name="CrossEntropyLoss"):
    loss_modul = importlib.import_module("torch.nn")
    loss_obj = getattr(loss_modul, loss_name)
    return loss_obj


def save_model(epoch, model, optimizer, train_loss, lr_scheduler, save_path, run, **kwargs):
    save_dict = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': train_loss,
                 'lr_scheduler': lr_scheduler.state_dict(),
                 }
    for key, value in kwargs.items():
        save_dict[key] = value
    torch.save(save_dict, save_path)


def get_lr_scheduler(policy, num_epochs, optimizer, trainloader, batch_size):
    # set learning rate scheduler
    if policy == 'identity':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: x
        )
    if policy == 'linear':
        # lr_scheduler adapted from
        # https://github.com/pytorch/vision/blob/f04e9cb9b6cc36afbd8694f19711224a23d28bba/references/segmentation/train.py#L136
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (len(trainloader) * batch_size * num_epochs)) ** 0.9
        )
    if policy == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    return lr_scheduler


def load_checkpoint(config, model, optimizer, lr_scheduler, device="cpu"):
    checkpoint = torch.load(config.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    if start_epoch > config.epochs:
        print(f"start_epoch ({start_epoch}) is greater than the number of total epochs ({config.epochs})")
        exit()
    return start_epoch
