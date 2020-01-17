import torch.optim as optim
import re
from neural_net import NeuralNet

EMBEDDINGS = ["", "\s", "\w[A-Za-z'-]+|[A-Z]", "\,", "\"", "\."]
EMBEDDINGINS_COUNT = len(EMBEDDINGS)

def embedLine(line:str):
    line = re.sub(EMBEDDINGS[2], "2", line)
    line = re.sub(EMBEDDINGS[1], "1", line)
    line = re.sub(EMBEDDINGS[3], "3", line)
    line = re.sub(EMBEDDINGS[4], "4", line)
    line = re.sub(EMBEDDINGS[5], "5", line)
    return line

def embeddedLineToTensor(embedded_line : str, batch_sz : str):
    tensor = torch.zeros(len(embedded_line), 1, EMBEDDINGS_COUNT)
    for i, letter in enumerate(embedded_line):
        tensor[i][0][int(letter)] = 1
    return tensor

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

net = NeuralNet()

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')