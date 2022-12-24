import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Define the neural network architecture
class PayloadGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PayloadGenerator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Load the SQLMap CSV file
df = pd.read_csv('payloads.csv')

# Extract the payloads and labels from the DataFrame
payloads1 = df['payload'].tolist()
print(payloads1)
payloads = []
chara= []
for payload in payloads1:
    pay = hash(payload)
    payloads.append(pay)


labels = df['label'].tolist()

# Convert the payloads and labels to tensors
payloads = torch.tensor(payloads, dtype=torch.int64)
labels = torch.tensor(labels, dtype=torch.bool)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Create a dataset from the payloads and labels
dataset = torch.utils.data.TensorDataset(payloads, labels)

# Define a dataloader to load the data in batches
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Define the model
model = PayloadGenerator(input_size=len(payloads[0]), hidden_size=64, output_size=2)

# Train the model
for epoch in range(10):
    for payload, label in dataloader:
        # Forward pass
        output = model(payload)
        loss = loss_fn(output, label)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for payload, label in dataloader:
        output = model(payload)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print(f'Accuracy: {correct / total:.4f}')

# Save the model
torch.save(model.state_dict(), 'model.pt')
