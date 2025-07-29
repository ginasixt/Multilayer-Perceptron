# PyTorch MLP for Diabetes Prediction

This project implements a **Multilayer Perceptron (MLP)** using PyTorch to predict the likelihood of diabetes based on health indicators.

The model is trained on the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).

---
## What I Learned

- How to design and implement an MLP in PyTorch
- What each layer and activation function does
- Why non-linearity is essential in deep learning
- How weights + ReLU allow neurons to specialize and respond to meaningful input patterns
- How sigmoid activation in the output provides interpretable probabilities
- How the training works

## Understanding the Internal Logic and Structure of the MLP
### Input Layer
- 21 input features (e.g., BMI, age, etc.)
- Not a “real layer” in PyTorch — just the raw input data

### 1st Hidden Layer
- 64 neurons
- Each neuron receives all 21 inputs
- In total: 64 neurons × 21 weights + 64 biases

### 2nd Hidden Layer
- 32 neurons
- Each neuron receives up to 64 inputs from the previous layer

### Output Layer
- 1 neuron
- Uses the sigmoid activation function to output a probability
- Returns a value between 0–1 representing the probability of diabetes being present

#### General Flow Between Layers
In each layer, every single neuron receives all outputs from the previous layer (dense connection) \
Each input is multiplied by the trainable weight, and a bias is added \
The resulting value (weighted sum + bias) is passed through an activation function (e.g., ReLU and for the Output Layer Sigmoid) \

## How ReLU Enables Learning
ReLU itself is simple:

- Outputs **0** if the input ≤ 0 --> the neuron does not influence the next layer  
- Outputs the input > 0 --> the neuron **fires** and passes the input trough

So applied to our neurons, if the **weighted sum of inputs**  
#TODO insert function 
exceeds the ReLU threshold (i.e., z > 0), the neuron becomes active (fires) and contributes to the next layer.

ReLU doesn't explicitly model logical rules like "high BMI + no exercise + age > 60",  
but it allows them to develop. 
The trained model adjusted the weights and biases so that only for such input combinations the activation threshold is crossed and the neuron fires.

Without ReLU (or another non-linear activation function), even deep networks are only a single linear transformation, incapable of modeling complex relationships.

## How the training works
The model is further optimized in each epoch cycle (i.e., a complete iteration over the training data set). The goal is to change the model parameters (weights and biases) so that the prediction comes closer and closer to the actual label.

The process is divided into five technical steps:



Through this process, the model learns to adjust the influence of each input feature (e.g. BMI, age, smoking) to improve prediction.

---
## Training Progress over Epochs
During training, the model goes through multiple epochs, full passes over the training data.
In each iteration:
- The model performs a forward pass, makes predictions, and calculates the loss
- Then it performs backpropagation, updating weights to minimize the loss
The log below shows how the binary cross-entropy loss decreased over 1000 training iterations,
indicating that the model is learning to better distinguish between diabetic and non-diabetic cases:

```text
Epoch 0, Loss: 0.7728
Epoch 100, Loss: 0.3241
Epoch 200, Loss: 0.3164
Epoch 300, Loss: 0.3142
Epoch 400, Loss: 0.3131
Epoch 500, Loss: 0.3124
Epoch 600, Loss: 0.3119
Epoch 700, Loss: 0.3115
Epoch 800, Loss: 0.3112
Epoch 900, Loss: 0.3110
```
---
## Evaluation
```text
Accuracy: 0.8679832860296437
Confusion Matrix:
 [[42846   893]
 [ 5805  1192]]
```

```text
Classification Report:
               precision    recall  f1-score   support

         0.0       0.88      0.98      0.93     43739
         1.0       0.57      0.17      0.26      6997

    accuracy                           0.87     50736
   macro avg       0.73      0.57      0.60     50736
weighted avg       0.84      0.87      0.84     50736
```

---
## The model still struggles with recall for diabetic cases. This could be improved by:

Adjusting class weights
Balancing the dataset
Trying more complex models or ensemble methods 
