# SplitLearning


### Introduction

In this research, I aimed to address the challenges in split learning, particularly focusing on optimizing communication overhead and latency while maintaining or improving model accuracy. 

### Literature Review

#### Split Learning
Split learning is a distributed training framework where the model is partitioned between the client and server, facilitating privacy-preserving machine learning.

#### Communication Overhead
Communication overhead refers to the extra time and resources required for data exchange between the client and server during split learning.

### The Model Used

#### The Forward Pass Function
The forward pass function calculates the output of the neural network by passing the input data through the layers.

#### ReLU Activation Function
ReLU (Rectified Linear Unit) activation function is used to introduce non-linearity in the model, defined as \( \text{ReLU}(x) = \max(0, x) \).

#### Cross Entropy Loss Function
The Cross Entropy Loss function measures the performance of a classification model whose output is a probability value between 0 and 1.

#### SGD Optimizer
Stochastic Gradient Descent (SGD) is an optimization algorithm used to minimize the loss function, updating weights using the formula: \( w = w - \eta \nabla L(w) \).

#### Quantization and Sparsification
Quantization reduces the number of bits required to represent gradients, while sparsification reduces the number of gradients by zeroing out small values, both aiming to reduce communication overhead.

### Research Method

#### The Three Scenarios

- **Baseline Model**: Implements standard split learning without optimizations.
- **Optimized Batch Model**: Increases the batch size to reduce communication frequency.
- **Compressed Model**: Applies gradient compression to reduce the communication volume.

#### Methodology

1. **Data Preparation**: Preprocess the MNIST dataset to standardize the data.
2. **Neural Network Architecture**: Split the model into client-side and server-side components.
3. **Split Learning Framework**: Implement split learning with communication protocols.
4. **Baseline Model Implementation**: Establish performance benchmarks.
5. **Optimizations**:
   - **Batch Size Optimization**: Increase batch size to reduce communication frequency.
   - **Gradient Compression**: Apply techniques to minimize communication volume.
6. **Experimental Evaluation**: Collect and analyze performance metrics.
7. **Result Interpretation**: Evaluate the effectiveness of optimizations.

### The Results and Discussion

#### Results

1. **Accuracy**: Analyze how model accuracy changes with different scenarios.
2. **Communication Overhead**: Measure the amount of data exchanged between client and server.
3. **Latency**: Evaluate the time taken for data exchange and processing.
4. **Computational Time**: Assess the total computation time required for training.

### Experimental Setup

#### Hardware Specifications
Experiments were conducted on a standard HP system and Google Colab environment with basic CPU runtime.

#### Software and Libraries
Utilized Python with libraries such as PyTorch (version 1.9), NumPy (version 1.21), and Matplotlib (version 3.4).

#### Dataset
MNIST dataset, a benchmark for handwritten digit classification.

### Discussion

- **Accuracy vs. Communication Overhead**: Balancing high accuracy with low communication overhead.
- **Latency vs. Computational Time**: Trade-offs between latency and overall computation time.
- **Scalability**: Assessing how the optimizations impact scalability in real-world applications.

### Conclusion

I restated my research aims and methodological approach, summarized key findings, evaluated theoretical and practical contributions, and recommended future research directions to further enhance split learning frameworks.

### Acknowledgment

I acknowledge the support of my college, IIIT Lucknow, and my supervisor, Dr. Mainak Adhikari of AAA college, for their guidance and support during this research project.
