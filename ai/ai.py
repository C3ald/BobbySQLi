import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pandas as pd
import base64
# Define the neural network model
class ModelClass(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
                super(ModelClass, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, output_size)
        def forward(self, x):
                x = self.fc1(x)
                x = self.fc1(x)
                return x
        
        def reward(self,r:int):
                for p in self.model.parameters():
                        p.data += r * p.grad

#To use PyTorch to train a model to generate its own SQL injection payloads using SQLMap data, you will need to follow these steps:
# Preprocess the SQLMap data:
# Load the CSV file containing the SQLMap data into a Pandas DataFrame.
# Extract the relevant columns from the DataFrame, such as the payloads and the labels indicating whether the payload was successful or not.
# Preprocess the data as needed, such as converting the payloads to integers using a vocabulary mapping.
# Split the data into training and testing sets.
# Define the model:
# Define a PyTorch model using the torch.nn module. The model should take in a sequence of integers representing the payload and output a prediction of whether the payload is successful or not. You may want to use a recurrent neural network (RNN) or a transformers-based model for this task.
# Train the model:
# Define a loss function that rewards the model for correctly predicting the success or failure of a payload.
# Define an optimizer to update the model's weights based on the loss.
# Use the torch.utils.data module to wrap the training and testing datasets in PyTorch DataLoaders.
# Use the model.train() method to put the model in training mode.
# Loop over the training DataLoader and compute the loss on each batch of data. Use the optimizer to update the model's weights based on the loss.
# Use the model.eval() method to put the model in evaluation mode.
# Loop over the testing DataLoader and compute the loss on each batch of data.
# Generate payloads:
# Use the model.eval() method to put the model in evaluation mode.
# Use the model to generate a payload by providing a seed sequence and using the model to predict the next tokens in the sequence. You can do this using the model.forward() method and passing in the seed sequence as the input.
# Continue generating tokens until the model reaches the end of the sequence or generates a special end-of-sequence token.
# Convert the generated sequence of integers back into a payload string using the vocabulary mapping.

                
                
                
def read_csv_and_convert_to_dataset(file_name) -> {'dataset':torch.utils.data.DataLoader,'length':int}:
        
        df = pd.read_csv(file_name, usecols=['payload', 'label'])
        payloads1 = df['payload'].tolist()
        labels = df = df['label'].tolist()
        payloads = []
        for payload in payloads1:
                fin = payload.encode()
                fin = int.from_bytes(fin, 'big')                
                payloads.append(fin)
        length = len(payloads)
        payloads = torch.tensor(payloads, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        tensor_data = torch.utils.data.Dataset(payloads, labels)
        dataset = torch.utils.data.DataLoader(tensor_data, batch_size=64, shuffle=True)
        return {'dataset':dataset, 'length':length}



def define_model(payload_list_length):
        model = ModelClass(input_size=payload_list_length, hidden_size=64, output_size=2)
        return model
        


def train_and_test(model,dataset):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        for epoch in range(15):
                for payload, label in dataset:
                        output = model(payload)
                        loss = loss_fn(output,label)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                print(f'Epoch: [{epoch+1}/10], Loss: {loss.item():.4f}')
        with torch.no_grad():
                correct = 0
                total = 0
                for payload, label in dataset:
                        output = model(payload)
                        _,predicted = torch.max(output.data, 1)
                        total += label.size(0)
                        correct += (predicted == label).sum().item()

    
# # Create an instance of the SQLMapDataset class
# dataset = SQLMapDataset(tensor_data)

# # Create a DataLoader instance to load the data in batches
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# # Iterate over the dataloader to retrieve the data
# for batch in dataloader:
#     # Process the data in the batch
#     print(batch)
