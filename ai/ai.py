import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pandas as pd
import base64
from torch import tensor
import itertools
import random
# Define the neural network model
class ModelClass(nn.Module):
        def __init__(self, input_size=95622, hidden_size=256, num_layers=8, output_size=random.randint(3,95622), device='cpu'):
                super(ModelClass, self).__init__()
                self.device = device
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.rnn = nn.GRU(input_size, hidden_size, num_layers)
                self.fc = nn.Linear(hidden_size, output_size)
  
        def forward(self, input, hidden):
                output, hidden = self.rnn(input, hidden)
                output = self.fc(output)
                return output, hidden
        
        def init_hidden(self):
                return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
        
        def reward(self,r:int):
                for p in self.model.parameters():
                        p.data += r * p.grad
padding_char = '\0'
     
def get_data():
        file = open('./payloads/all.txt','r',errors='ignore')
        training_data = file.readlines()
        final = []
        for t in training_data:
                final.append(t.strip())
        training_data = final
        longest = len(max(training_data, key=len))
        shortest = len(min(training_data, key=len))
        char_to_int = {c: i for i, c in enumerate(sorted(set(''.join(training_data))))}
        training_data = [[char_to_int[c] for c in string] for string in training_data]
        original = final

        return {'data': training_data, 'length': len(char_to_int), 'long': longest, 'short': shortest, 'original': original}
def initialize():
        data = get_data()
        input_size = 95622
        print(input_size)
        hidden_size = 256
        num_layers = 8
        minimum = data['short']
        maximum = data['long']
        output_size = random.randint(minimum,maximum)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = ModelClass(input_size, hidden_size, num_layers, output_size, device)
        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.MSELoss
        
        return model, optimizer, loss_fn

def int_train_model():
        data = get_data()
        max_length = data['long']
        original = data['original']
        shortest = data['short']
        data['data'] = list(data['data'])
        fin = []
        for lis in data['data']:
                for li in lis:
                        fin.append(li)
        data['data'] = fin

        # Define the labels
        labels = []
        for x in range(len(data['data'])):
                labels.append(bool(random.randint(0,1)))
        
        # Convert the labels list to a tuple
        
        # Create a PyTorch tensor from the labels and data
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data['data'] = [labels,data['data']]
        model, optimizer, loss_fn = initialize()
        num_epochs = 50
        batch_size = 32
        label = data['data'][0]
        inputs = data['data'][1]
        inputs = torch.tensor(inputs,dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        inputs = inputs.view(-1, 1,95622)
        labels = labels.view(-1, 1,95622)
        print(inputs.dtype, labels.dtype)
        
        model.train()
        for epoch in range(num_epochs):
                hidden = model.init_hidden()
                
                for input, label in zip(inputs,labels):
                        optimizer.zero_grad()
                        
                        output, hidden = model(inputs, hidden)
                        optimizer.step()
                        # output = convert_ints(output, original)
                print(f'Epoch {epoch+1}/{num_epochs}, output: {output.max()}')

def convert_ints(output, original_training_data):
        int_to_char = {c: i for i, c in enumerate(sorted(set(''.join(original_training_data))))}
        int_to_char = {i: c for c, i in int_to_char.items()}
        output_text = ""
        for i in output:
                output_text += int_to_char[i]
        return output_text

def generate_payload(model,original_data, start_char=' ', length=100):
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize the hidden state
    hidden = model.init_hidden()
    
    # Initialize the input character and the generated payload list
    input_char = start_char
    payload = []
    
    # Loop until the payload reaches the desired length
    while len(payload) < length:
        # Convert the input character to a tensor and pass it through the model
        input_tensor = torch.tensor(char_to_int[input_char], dtype=torch.float).view(1, 1, -1)
        output, hidden = model(input_tensor, hidden)
        
        # Convert the output tensor to a probability distribution
        output_dist = nn.Softmax(dim=2)(output)
        
        # Sample a character index from the probability distribution
        char_index = torch.multinomial(output_dist, num_samples=1).item()
        
        # Convert the character index to a character
        char_to_int = {c: i for i, c in enumerate(sorted(set(''.join(original_data))))}
        char = int_to_char[char_index]
        
        # Append the character to the generated payload list
        payload.append(char)
        
        # Set the input character to the newly generated character
        input_char = char
    
    # Return the generated payload as a string
    return ''.join(payload)

#int_train_model()

# def int_train_model():
#         data = get_data()
#         max_length = data['long']
#         shortest = data['short']
#         data['data'] = list(data['data'])
#         fin = []
#         for lis in data['data']:
#                 for li in lis:
#                         fin.append(li)
#         data['data'] = fin
#         labels = []
#         for x in range(len(data['data'])):
#                 labels.append(x)
        
#         #labels = tuple(labels)
        
#         # data['data'] = torch.tensor((labels,data['data']),dtype=torch.long)
#         data['data'] = torch.tensor(data['data'],dtype=torch.long)
#         labels = torch.tensor(labels,dtype=torch.long)
#         data['data'] = data['data'].view(-1,1)
#         labels = labels.view(-1,1)
#         labels = labels.to('cuda')
#         model, optimizer, loss_fn = initialize()
#         #data['data'] = torch.stack((labels, data['data']), dim=1)
#         print(data['data'])
#         model, optimizer, loss_fn = initialize()
#         num_epochs = 50
#         batch_size = 32

#         model.train()
#         for epoch in range(num_epochs):
#                 hidden = model.init_hidden()
#                 for input, label in data['data']:
#                         optimizer.zero_grad()
#                         input = input.view(-1, 1).to(device)
#                         label = label.view(-1, 1).to(device)
#                         output, hidden = model(input, hidden)
#                         loss = loss_fn(output, label)
#                         loss.backward()
#                         optimizer.step()
#                 print(f'Epoch {epoch+1}/{num_epochs}: Loss {loss.item():.4f}')
        
if __name__ == '__main__':
        data = get_data()
        original = data['original']
        generate_payload(model=ModelClass(),original_data=original)