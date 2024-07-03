# Usage

Directory osmDir is used to store osm files, jsonDir is used to store rc and lc extracted from osm and joint's node information.

1. use downloadLatestByNeighboring.py to download osm data to osmDir
2. use getNodeJson.py to import the node information of the osm file into jsonDir.
3. use writeToCSV to import the information between the maps into CSV.
4. use train.py to train the model and generate the weights.
5. use detect.py file to import the weight file and make predictions on the CSV data

## Network structure

1. **Input layer (self.linear1)**: this layer is like the doorway to the factory, it receives 8 features as input. You can think of these features as raw materials that need to be further processed before they can be used in the final prediction.
2. **First Hidden Layer (self.linear1 + ReLU activation function)**: the raw materials will first arrive here when they enter the factory. There are 6 neurons in this layer, so you can think of it as 6 different processing equipment. Each device performs a certain deformation operation on the raw material and adds nonlinearity using the ReLU function. This is equivalent to the initial processing of the raw material.
3. **The second hidden layer (self.linear2 + ReLU activation function)**: this layer receives the output from the first hidden layer, which is the initially processed product, and performs further processing. This layer has 4 neurons representing 4 different processing devices. Like the first hidden layer, each device performs certain deformation operations on the product and applies the ReLU function again to add nonlinearity.
4. **Output Layer (self.linear3 + Sigmoid Activation Function)**: this is the last process, where the final processing of our product is done. There is only one neuron in this layer, which transforms the output of the previous layer into a value between 0 and 1. This value is our final prediction. The use of the Sigmoid function ensures that the output is in the range (0, 1), which is suitable for representing probability.

Translated with DeepL.com (free version)