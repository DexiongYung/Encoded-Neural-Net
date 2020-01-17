from torch import nn
class NeuralNetwork(nn.Module):
    def __init__(self, input_sz : int, hidden_sz : int, output_sz : int):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(input_sz, hidden_sz)
        # Output layer
        self.output = nn.Linear(hidden_sz, output_sz)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x