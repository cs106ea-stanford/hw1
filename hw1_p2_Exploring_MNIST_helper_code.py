### SETUP

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import random
import time
from datetime import datetime

try:
    import google.colab
    running_on_colab = True
except ImportError:
    running_on_colab = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


### STANDARD IPYWIDGET IMPORTS

import ipywidgets as widgets
from ipywidgets import HBox, VBox, HTML, Button, Output

from IPython.display import clear_output

from typing import Dict

### SETUP CSS STYLES
#   These will only be used if you if you explicitly include
#   them in a given "display()" call

html_style = HTML(
    value="""
<style>
.control-major-label {
    font-size: 1.2em;
    font-weight: bold;
}
.control-label {
    font-size: 1em;
    font-weight: bold;
}
.control-minor-label {
    font-size: 0.9em;
}
.widget-checkbox {
    width: auto !important;  /* Adjust this if necessary */
    /*border: 1px solid blue;*/ /* To see the actual space taken by the checkbox container */
}
.widget-checkbox > label {
    margin: 0 !important;
    padding: 0 !important;
    width: auto !important;
    /*border: 1px solid red;*/ /* To see the space taken by the label */
}
.widget-checkbox input[type="checkbox"] {
    margin: 0 !important;
}
.widget-inline-hbox .widget-label {
    flex: 0 0 auto !important;
}
.widget-inline-hbox {
    align-items: center; /* Align items vertically in the center */
    min-width: 0; /* Helps prevent flex containers from growing too large */
}
.code {
    font-family: 'Courier New', Courier, monospace;
    font-weight: bold;
    line-height: 0.5;
    margin: 0;
    padding: 0;
}
</style>

    """
)

### LOAD DATASET

# mean and std_dev calculated from determine_mnist_weights.ipynb
# we will be using weights from that file, so reuse same mean and std_dev

mean = 0.13066048920154572
std_dev = 0.308107852935791

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std_dev)
])

NUMBER_OF_EPOCHS = 20
LEARNING_RATE = 0.02
BATCH_SIZE = 128

PRINT_RATE = 1  # how often we should print results

def load_dataset():
    global training_images, testing_images, training_images_length, testing_images_length, training_loader
    training_images = datasets.MNIST("MNIST_data", transform=transform,
                                            download=True,train=True)
    testing_images = datasets.MNIST("MNIST_data", transform=transform,
                                            download=True,train=False)
    training_images_length = len(training_images)
    testing_images_length = len(testing_images)

    training_loader = DataLoader(training_images,
                                   batch_size=BATCH_SIZE, shuffle=True)

# Create button and output widget
explore_display_samples_button = Button(description="Display More Samples", button_style='info')
explore_samples_output = Output()

def regenerate_explore_samples_data(_):
    ROWS = 4
    COLS = 4

    with explore_samples_output:
        explore_samples_output.clear_output(wait=True)

        fig, axs = plt.subplots(4, 4)
        for row in range(ROWS):
            for col in range(COLS):
                image, label = training_images[random.randint(0, training_images_length - 1)]
                denormalized_image = image.squeeze() * std_dev + mean
                axs[row][col].imshow(denormalized_image, cmap='gray')  # squeeze to remove unnecessary dimension
                axs[row][col].text(2, 2, str(label), color='yellow', fontsize=12, ha='left', va='top')
                axs[row][col].axis('off')
        
        plt.tight_layout()
        plt.show()

# Attach the function to the button click
explore_display_samples_button.on_click(regenerate_explore_samples_data)

# Display the button and output
# display(html_style,VBox([explore_display_samples_button, explore_samples_output]))

def display_init_explore_data():
    display(html_style,VBox([explore_display_samples_button,explore_samples_output]))
    regenerate_explore_samples_data(None)

# DEFINE MODEL

class MNISTBasicNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,network_input):
        output = self.layers(network_input)
        if not self.training:
            output = nn.functional.softmax(output, dim=1)
        return output
    


def determine_accuracy(model):
    test_loader = DataLoader(testing_images,
                                   batch_size=BATCH_SIZE, shuffle=False)

    total_correct = 0
    
    model.eval()
    with torch.no_grad():
        for img_batch, label_batch in test_loader:
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
    
            result_batch = model(img_batch)
            _, predicted_batch = torch.max(result_batch, 1)
    
            total_correct += (predicted_batch == label_batch).sum().item()

    total_count = len(test_loader.dataset)

    return (100 * total_correct / total_count, total_correct, total_count) 

def print_accuracy(model):
    accuracy, correct, total = determine_accuracy(model)
    print(f"Correctly Predicted: {correct:,}")
    print(f"Total Samples: {total:,}")
    print(f"Accuracy: {accuracy:.2f}%")

# CREATE MODEL AND LOAD WITH PRE-TRAINED WEIGHTS

def define_and_load_pre_trained_model():
    global pre_trained_model
    pre_trained_model = MNISTBasicNetwork()
    pre_trained_model.load_state_dict(torch.load("mnist_basic_network_parameters.pth", weights_only=True))
    pre_trained_model = pre_trained_model.to(device)
    print(pre_trained_model)

def determine_accuracy_pre_trained_model():
    print_accuracy(pre_trained_model)

def test_digits(model,output, n):
    with output:
        output.clear_output(wait=True)
        for i in range(n):
            image, label = testing_images[random.randint(0, testing_images_length - 1)]
            denormalized_image = image.squeeze() * std_dev + mean
            plt.figure(figsize=(2, 2))
            plt.imshow(denormalized_image, cmap='gray')
            plt.axis('off')
            plt.title(f"Label: {label}")
            plt.show()

            image = image.to(device)
            predictions = model(image.unsqueeze(0)).detach().cpu().numpy()

            for digit, percentage in enumerate(predictions[0] * 100):  # Use predictions[0] for a single image
                print(f"{digit}: {percentage:.1f}%")
                
def regenerate_trained_results(_):
    test_digits(pre_trained_model,pre_trained_results_output, 5)
    
pre_trained_results_button = Button(description="Display More Results", button_style='info')
pre_trained_results_output = Output()

# Attach the function to the button click
pre_trained_results_button.on_click(regenerate_trained_results)

def display_pre_trained_results():
    display(html_style,VBox([pre_trained_results_button,pre_trained_results_output]))

def find_problematic_samples(model,max_percent,n):
    """
    Searches for examples that the model is having difficulty with.  
    Creates a new DataLoader and randomizes so we get different results each time.

    Args:
        model: model to use
        max_percentage: the highest probability allowed for any label to be considered 'problematic'
        n: number to return

    Returns:
        list of (features, label) tuples
        list will be empty if none found matching criteria
    """
    return_list = []
    randomized_test_loader = DataLoader(testing_images,
                                   batch_size=BATCH_SIZE, shuffle=True)

    model.eval()
    with torch.no_grad():
        for feature_batch, label_batch in randomized_test_loader:
            feature_batch_device = feature_batch.to(device)
            predictions = model(feature_batch_device) 

            for i in range(predictions.size(0)):
                max_prob, _ = torch.max(predictions[i], dim=0)  # Find the highest probability
                if max_prob*100 <= max_percent:
                    return_list.append((feature_batch[i], label_batch[i]))

                # Stop collecting if we've found enough samples
                if len(return_list) >= n:
                    return return_list

    return return_list

def show_digits(output,model,samples_list):
    with output:
        output.clear_output(wait=True)
        
        for image, label in samples_list:
            denormalized_image = image.squeeze() * std_dev + mean
            plt.figure(figsize=(2, 2))
            plt.imshow(denormalized_image, cmap='gray')
            plt.axis('off')
            plt.title(f"Label: {label}")
            plt.show()

            image = image.to(device)
            predictions = model(image.unsqueeze(0)).detach().cpu().numpy()

            for digit, percentage in enumerate(predictions[0] * 100):  # Use predictions[0] for a single image
                print(f"{digit}: {percentage:.1f}%")

def show_problems(_):
    samples_list = find_problematic_samples(pre_trained_model,40,5)
    show_digits(pre_trained_problems_output, pre_trained_model,samples_list)

pre_trained_problems_button = Button(description="Display More Problems", button_style='info')
pre_trained_problems_output = Output()

# Attach the function to the button click
pre_trained_problems_button.on_click(show_problems)

def display_pre_trained_problems():
    display(html_style,VBox([pre_trained_problems_button,pre_trained_problems_output]))
    show_problems(None)

# CREATE NEW RANDOM MODEL

def create_new_model(_):
    global working_model_0_epochs
    working_model_0_epochs = MNISTBasicNetwork()
    working_model_0_epochs = working_model_0_epochs.to(device)
    with create_new_epoch0_output:
        create_new_epoch0_output.clear_output(wait=True)
        print(f"Model Reset at {datetime.now().strftime('%I:%M:%S %p')}")
        
create_new_epoch0_button = Button(description="Reset Model", button_style='info')
create_new_epoch0_output = Output()

# Attach the function to the button click
create_new_epoch0_button.on_click(create_new_model)

def display_create_new_model():
    display(html_style,VBox([create_new_epoch0_button,create_new_epoch0_output]))
    create_new_model(None)

def determine_accuracy_init_model():
    print_accuracy(working_model_0_epochs)

def regenerate_results_0(_):
    test_digits(working_model_0_epochs,working_model_0_results_output, 5)
    
working_model_0_results_button = Button(description="Display More Results", button_style='info')
working_model_0_results_output = Output()

# Attach the function to the button click
working_model_0_results_button.on_click(regenerate_results_0)

def display_working_model_0_results():
    display(html_style,VBox([working_model_0_results_button,working_model_0_results_output]))
    regenerate_results_0(None)

import copy

def duplicate_model(original_model):
    # Create a new instance of the model
    new_model = MNISTBasicNetwork()
    
    # Load the state dictionary of the original model
    new_model.load_state_dict(copy.deepcopy(original_model.state_dict()))
    
    # Move the new model to the same device as the original model
    new_model = new_model.to(next(original_model.parameters()).device)
    
    return new_model

def run_epochs(previous_model, output, epoch_count, n=5):

    """
    Duplicate the previous_model, run n training epochs, and return the new model.
    Print results to the output widget. The first epoch run will be epoch_count + 1.

    Args:
        previous_model: The model to duplicate and train.
        output: Output widget for displaying training logs.
        epoch_count: The last completed epoch count. Training starts from epoch_count + 1.
        n: Number of epochs to train (default is 5).

    Returns:
        new_model: The trained model.
    """
    
    new_model = duplicate_model(previous_model)

    with output:
        output.clear_output(wait=True)

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(new_model.parameters(),lr = LEARNING_RATE)
        new_model.train()
        
        start_time = time.time()
        
        for epoch in range(epoch_count + 1, epoch_count + n + 1):
            total_instances = 0
            total_loss = 0
            new_model.train()
            for img_batch, label_batch in training_loader:
                img_batch, label_batch = img_batch.to(device), label_batch.to(device)
                result_batch = new_model(img_batch)
                loss = loss_func(result_batch, label_batch)
        
                mini_batch_size = len(img_batch)
                total_instances += mini_batch_size
                total_loss += mini_batch_size * loss.item()
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            if epoch % PRINT_RATE == 0:
                print(f"epoch: {epoch}, loss: {(total_loss/total_instances):.4f}")
        
        end_time = time.time()
        print(f"Training time: {(end_time - start_time):.2f} seconds")

    return new_model
    
def train_to_5(_):
    global working_model_5_epochs
    working_model_5_epochs = run_epochs(working_model_0_epochs, working_model_5_train_output, 0, 5)
    
working_model_5_train_button = Button(description="Retrain Network", button_style='info')
working_model_5_train_output = Output()

# Attach the function to the button click
working_model_5_train_button.on_click(train_to_5)

def display_working_model_5_train():
    display(html_style,VBox([working_model_5_train_button,working_model_5_train_output]))
    train_to_5(None)

def determine_accuracy_model_5_epochs():
    print_accuracy(working_model_5_epochs)

def regenerate_results_5(_):
    test_digits(working_model_5_epochs,working_model_5_results_output, 5)
    
working_model_5_results_button = Button(description="Display More Results", button_style='info')
working_model_5_results_output = Output()

# Attach the function to the button click
working_model_5_results_button.on_click(regenerate_results_5)

def display_working_model_5_results():
    display(html_style,VBox([working_model_5_results_button,working_model_5_results_output]))
    regenerate_results_5(None)
    
def train_to_10(_):
    global working_model_10_epochs
    working_model_10_epochs = run_epochs(working_model_5_epochs, working_model_10_train_output, 5, 5)
    
working_model_10_train_button = Button(description="Retrain Network", button_style='info')
working_model_10_train_output = Output()

# Attach the function to the button click
working_model_10_train_button.on_click(train_to_10)

def display_working_model_10_train():
    display(html_style,VBox([working_model_10_train_button,working_model_10_train_output]))
    train_to_10(None)

def determine_accuracy_model_10_epochs():
    print_accuracy(working_model_10_epochs)

def regenerate_results_10(_):
    test_digits(working_model_10_epochs,working_model_10_results_output, 5)
    
working_model_10_results_button = Button(description="Display More Results", button_style='info')
working_model_10_results_output = Output()

# Attach the function to the button click
working_model_10_results_button.on_click(regenerate_results_10)

def display_working_model_10_results():
    display(html_style,VBox([working_model_10_results_button,working_model_10_results_output]))
    regenerate_results_10(None)
    
def train_to_15(_):
    global working_model_15_epochs
    working_model_15_epochs = run_epochs(working_model_10_epochs, working_model_15_train_output, 10, 5)
    
working_model_15_train_button = Button(description="Retrain Network", button_style='info')
working_model_15_train_output = Output()

# Attach the function to the button click
working_model_15_train_button.on_click(train_to_10)

def display_working_model_15_train():
    display(html_style,VBox([working_model_15_train_button,working_model_15_train_output]))
    train_to_15(None)

def determine_accuracy_model_15_epochs():
    print_accuracy(working_model_15_epochs)