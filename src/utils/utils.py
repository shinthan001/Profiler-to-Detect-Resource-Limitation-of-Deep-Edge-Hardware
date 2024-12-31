import csv
import sys, os

import torch
from torchvision import transforms

def append_csv(path, new_row):

    try:
        # Open the CSV file in append mode and write the new row
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)
        print('Row appended and saved successfully.')
    except:
        print(f"Unable to save in {path}")


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def create_raw_PIL_images(batch, width, height):

    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage()
    ])
    images = [transform(torch.zeros(3, width+20, height+20)) for _ in range (batch)]
    
    return images