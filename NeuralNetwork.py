# LINK TO NEURAL NETWORK: https://hmkcode.com/images/ai/bp_forward_2.png

# Import  Packages
import random
import matplotlib.pyplot as plt

# Get dataset
x = [[2, 3], [3, 4], [6, 8], [2, 6]]
y = [1, 1, 2, 4] # y values are x[0] - x[1]

# Weight lists (for visualizations)
w1l = []
w2l = []
w3l = []
w4l = []
w5l = []
w6l = []

# Loss list (for visualizations)
loss = []

# Initialize first layer weights(2 input neurons • 2 hidden layer neurons = 4 weights [assuming densely connected layers])
w1 = random.randint(0, 100) / 100
w2 = random.randint(0, 100) / 100
w3 = random.randint(0, 100) / 100
w4 = random.randint(0, 100) / 100

# Hidden layer (2 hidden neurons • 1 output neuron = 2 weights [assuming densely connected layers]) 
w5 = random.randint(0, 100) / 100
w6 = random.randint(0, 100) / 100

# Set learning rate
lr = 0.05

# Pass inputs through the network
def forward_pass(w1, w2, w3, w4, w5, w6, xs, ys):

  # Calculate neuron values
  neuron1 = w1 * xs[0] + w2 * xs[1] # w1 and w2 are each attached to different x neurons but the same hidden layer neuron: neuron1
  neuron2 = w3 * xs[0] + w4 * xs[1] # w3 and w4 are each attached to different x neurons but the same hidden layer neuron: neuron2

  prediction = neuron1 * w5 + neuron2 * w6 # w5 and w6 are attached to different hidden layer neurons (neuron1 and neuron2 respectively) but the same the y neuron
  error =  prediction - ys

  return neuron1, neuron2, prediction, error

# Update the weights 
def backprop(w1, w2, w3, w4, w5, w6, xs, neuron1, neuron2, lr, error):
  delta = lr * error

  # Reassign weights starting from the front of the NN (hence BACKpropogation)
  nw6 = w6 - delta * neuron2 # neuron2 since w6 is attached to that hidden neuron 
  nw5 = w5 - delta * neuron1 # neuron1 since w5 is attached to that hidden neuron

  nw4 = w4 - delta * xs[1] * w6 # x[1] because w4 is connected to the second x neuron and w6 because w4 and w6 are attached to the same hidden neuron
  nw3 = w3 - delta * xs[0] * w6 # x[0] because w3 is connected to the first x neuron and w6 because w3 and w6 are attached to the same hidden neuron
  nw2 = w2 - delta * xs[1] * w5 # x[1] because w2 is connected to the second x neuron and w5 because w2 and w5 are attached to the same hidden neuron
  nw1 = w1 - delta * xs[0] * w5 # x[0] because w1 is connected to the first x neuron and w5 because w1 and w5 are attached to the same hidden neuron
  
  return nw1, nw2, nw3, nw4, nw5, nw6

# Train the network
i = 0
epochs = 1000

for epoch in range(epochs):
  if i > len(x) - 1:
    i = 0

  # View epoch and corresponding weights
  print("\nEpoch", epoch, "Weights:", w1, w2, w3, w4, w5, w6)
  
  # Add the weights to lists for visualization
  w1l.append(w1)
  w2l.append(w2)
  w3l.append(w3)
  w4l.append(w4)
  w5l.append(w5)
  w6l.append(w6)

  neuron1, neuron2, prediction, error = forward_pass(w1, w2, w3, w4, w5, w6, x[i], y[i])
  
  # Evaluation model
  print("Predictions:", prediction, "Error:", error)
  loss.append(error)

  # Get new weights
  w1, w2, w3, w4, w5, w6 = backprop(w1, w2, w3, w4, w5, w6, x[i], neuron1, neuron2, lr, error)

  i += 1

# Get predictions
input_value = [5, 3] # Change this value to have the model predict different values
print("\n\n", forward_pass(w1, w2, w3, w4, w5, w6, input_value, 1)[3]) # Forward passing returns predictions

# Visualize Weights
plt.plot(w1l, label = 'Weight 1')
plt.plot(w2l, label = 'Weight 2')
plt.plot(w3l, label = 'Weight 3')
plt.plot(w4l, label = 'Weight 4')
plt.plot(w5l, label = 'Weight 5')
plt.plot(w6l, label = 'Weight 6')
plt.xlabel('Epochs')
plt.ylabel('Weight Value')
plt.legend()

plt.show()

print("\n\n")

# Visualize loss (error)
plt.plot(loss, label = 'Error')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
