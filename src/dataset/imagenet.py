import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_transforms(image_size=(256,256)):

    # Define data transformations for ImageNet dataset
    return transforms.Compose([
        transforms.Resize(image_size),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_data(path, batch_size, image_size=(256,256)):

    # Define data transformations for ImageNet dataset
    transform = get_transforms(image_size)

    # Load ImageNet validation dataset
    imagenet_data = datasets.ImageNet(root=path, split='val', transform=transform)

    # Create a DataLoader for the ImageNet dataset
    dataloader = DataLoader(imagenet_data, batch_size=batch_size, shuffle=True, num_workers=torch.get_num_threads())
    return dataloader

# Function to calculate top-k accuracy
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate_accuracy(model, dataloader):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    top1_correct, top5_correct = 0, 0
    it = iter(dataloader)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(len(dataloader)), desc="Validating.."):
            images, labels = next(it)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # get top1 and top5 accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1,5))
            top1_correct += acc1[0]
            top5_correct += acc5[0]


    # Calculate top-1 and top-5 accuracies
    top1_accuracy = top1_correct / len(dataloader)
    top5_accuracy = top5_correct / len(dataloader)

    print('Top-1 Accuracy on ImageNet validation set: {:.2f}%'.format(top1_accuracy))
    print('Top-5 Accuracy on ImageNet validation set: {:.2f}%'.format(top5_accuracy))
    return top1_accuracy, top5_accuracy
    

