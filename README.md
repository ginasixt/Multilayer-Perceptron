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

So applied to our neurons, if the **weighted sum of inputs** \
$`w_1 x_1 + w_2 x_2 + ... + b`$ \
exceeds the ReLU threshold (i.e., z > 0), the neuron becomes active (fires) and contributes to the next layer.

ReLU doesn't explicitly model logical rules like "high BMI + no exercise + age > 60",  
but it allows them to develop. 
The trained model adjusted the weights and biases so that only for such input combinations the activation threshold is crossed and the neuron fires.

Without ReLU (or another non-linear activation function), even deep networks are only a single linear transformation, incapable of modeling complex relationships.

## How the training works
The model is further optimized in each epoch cycle (i.e., a complete iteration over the training data set). The goal is to change the model parameters (weights and biases) so that the prediction comes closer and closer to the actual label.
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

### The process is divided into five steps:
Through this process, the model learns to adjust the influence of each input feature (e.g. BMI, age, smoking) to improve prediction.

#### 1. Forward Pass
```python
outputs = model(X_train)
```

The input matrix X_train (size: [n_samples, n_features]) is propagated through the network. \
Each layer performs: \
$`z = XW + b`$ \
where W is the weight matrix and b is the bias vector.\
A non-linear activation function (e.g., ReLU) is applied after each layer to allow complex relationships to emerge. \
The output layer applies a Sigmoid function to return probabilities between 0 and 1 for each patient.

#### 2. Compute Loss (Error)
```python
loss = criterion(outputs, y_train)
```
The Binary Cross Entropy Loss compares the predicted probabilities with the actual labels (y_train). \
The final loss is the mean over all samples in the batch

#### 3. Zero out Gradients
```python
optimizer.zero_grad()
```
PyTorch accumulates gradients by default.
This step clears previous gradients to avoid mixing them with the current batch.

#### 4. Backpropagation (Gradient Computation)
```python
loss.backward()
```
The `.backward()` function in PyTorch:
  - Automatically **traces the entire computation graph** from output back to input
  - Applies the **chain rule** to calculate the **gradient (slope)** for every parameter
  - These gradients are stored and will be used in the next step
 
These gradients indicate the direction to update each weight to reduce the loss.
PyTorch automatically calculates the gradient of the loss with respect to all parameters.

#### 5. Parameter Update
```python
optimizer.step()
```
After the backward pass, every parameter (weight or bias) has a **gradient**, the value that tells the model how much that parameter contributed to the error. \
For example a positive gradient means increasing this weight increases the error.
**SGD (Stochastic Gradient Descent)** uses these gradients to update each parameter slightly in the direction that reduces the loss. \
This adjustment is called a learning step.  
It’s controlled by a hyperparameter called the **learning rate** (often written as `lr`). 

**Example:** \
A weight currently has the value `0.5`, and its gradient is `+0.2`. \
If the learning rate is `0.01`, the new weight becomes: \
$`0,5 - 0,01 * 0,2 = 0,498`$ \
we decrease the weight slightly. So the optimizer nudges the weight in the right direction.  \
This is repeated for every parameter, every epoch.

## Evaluation
```text
Accuracy: 0.8679832860296437
Confusion Matrix:
 [[42846   893]
 [ 5805  1192]]
```
- True Negatives (no diabetes, predicted correctly): 42,846
- False Positives (no diabetes, predicted as yes): 893
- False Negatives (has diabetes, missed): 5,805
- True Positives (has diabetes, detected): 1,192

```text
Classification Report:
               precision    recall  f1-score   support

         0.0       0.88      0.98      0.93     43739
         1.0       0.57      0.17      0.26      6997

    accuracy                           0.87     50736
   macro avg       0.73      0.57      0.60     50736
weighted avg       0.84      0.87      0.84     50736
```
- Precision (1.0): 57% of predicted diabetics were actually diabetic
- Recall (1.0): Only 17% of actual diabetics were correctly detected
- F1-Score: shows poor diabetic detection performance

### Conclusion
The model still struggles with recall for diabetic cases. This could be improved by:
- Adding more layers or neurones, but with being aware for overfitting
- Adding more iterations, watching the Loss Log
- The dataset is unbalanced (more non-diabetics)
   - Adjusting pos_weight in the loss function
   - Oversampling of the minority class (SMOTE, RandomOverSampler)
   - Using focal loss instead of BCE to focus on difficult cases
- Using different optimizer like Adam and adjusting the learning rate
- weigt decay or dropout
