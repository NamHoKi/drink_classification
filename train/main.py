import copy
import os.path
import os
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from customdataset import my_dataset
from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.09,
                            rotate_limit=25, p=0.6),
        A.Resize(width=224, height=224),
        A.RandomBrightnessContrast(p=0.6),
        A.VerticalFlip(p=0.6),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(width=224, height=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(width=224, height=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_dataset = my_dataset("./dataset/train/", transform=train_transform)
    val_dataset = my_dataset("./dataset/val/", transform=val_transform)
    test_dataset = my_dataset("./dataset/test/", transform=test_transform)

    def visulize_augmentations(dataset, samples=4, cols=2):
        dataset = copy.deepcopy(dataset)
        dataset.transform = A.Compose([t for t in dataset.transform
                                        if not isinstance(
                t, (A.Normalize, ToTensorV2)
            )])
        rows = samples // cols
        _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
        for i in range(samples):
            image, _ = dataset[10]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()

    # test = visulize_augmentations(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


    # model = models.vgg19()
    # model.head = nn.Linear(in_features=4096, out_features=6)
    # model.to(device)
    # net = models.efficientnet_b0(pretrained=True)
    # net.classifier[1] = nn.Linear(in_features=1280, out_features=3)
    # net.to(device)
    # model = torchvision.models.resnet34(pretrained=True)
    # model.fc = torch.nn.Linear(in_features=512, out_features=6)
    # model.to(device)
    model = torch.hub.load('facebookresearch/deit:main',
                           'deit_tiny_patch16_224', pretrained=False)
    model.head = nn.Linear(in_features=192, out_features=6)
    model.to(device)

    loss_function = LabelSmoothingCrossEntropy()
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  # net의 파라메타를 넣어줘야 한다.
    # lr scheduler
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30,
                                                       gamma=0.1)
    epochs = 20

    best_val_acc = 0.0

    train_steps = len(train_loader)
    val_steps = len(val_loader)
    save_path = "./best_resnet34_4.pt"
    dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                                    columns=["Epoch", "Accurascy"])
    if os.path.exists(save_path):
        best_val_acc = max(pd.read_csv("./modelAccuracy4.csv")["Accuracy"].tolist())

    for epoch in range(epochs):
        runing_loss = 0
        val_acc = 0
        train_acc = 0

        # net.train()
        model.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour='green')
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = loss_function(outputs, labels)

            scheduler.step()
            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            runing_loss += loss.item()

            train_bar.desc = f"train epoch [{epoch + 1}/{epochs}], loss >> {loss.data:.3f}"

        model.eval()
        with torch.no_grad():
            val_loss = 0
            valid_bar = tqdm(val_loader, file=sys.stdout, colour='red')
            for data in valid_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
                val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

        val_accuracy = val_acc / len(val_loader)
        train_accuracy = train_acc / len(train_loader)

        dfForAccuracy.loc[epoch, "Epoch"] = epoch + 1
        dfForAccuracy.loc[epoch, "Train_Accuracy"] = round(train_accuracy, 3)
        dfForAccuracy.loc[epoch, "Train_Loss"] = round(runing_loss / train_steps, 3)
        dfForAccuracy.loc[epoch, "Val_Accuracy"] = round(val_accuracy, 3)
        dfForAccuracy.loc[epoch, "Val_Loss"] = round(val_loss / val_steps, 3)
        print(f"epoch [{epoch+1}/{epochs}] trian_loss{(runing_loss / train_steps):.3f} train_acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), save_path)

        if epoch == epochs - 1:
            dfForAccuracy.to_csv("./modelAccuracy4.csv", index=False)

    torch.save(model.state_dict(), "./best_deit.pt")


if __name__ == '__main__':
    main()

