# Awesome Convolutional Neural Networks

A list of papers I used for my thesis about convolutional neural networks with a focus on batch normalization. The papers are mostly ordered chronologically in terms of their publication date. I divided the the papers in sections [Early word](#early-word), [Pre-AlexNet](#pre-alexnet), [Post-AlexNet](#post-alexnet), and [Batch Normalization](#batch-normalization).

## Early work

##### [Learning Representations by Backpropagating Errors](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) (Rumelhart et al 1986)

Invention of the *backpropagation* algorithm.

##### [Backpropagation Applied to Handwritten Zip Code Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf) (LeCun et al 1989)

Backpropagation applied to character recognition with *weight sharing*.

##### [Handwritten Digit Recognition with a Backpropagation Network](http://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf) (LeCun et al 1990)

First paper on *convolutional neural networks* trained with backpropagation.

##### [Gradient-based Learning Applied to Document Recogintion](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (LeCun et al 1998)

Overview of training end-to-end systems such as convolutional neural networks with gradient-based optimization.

##### [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (LeCun et al 1998)

Gives many practical recommendations for traning multi-layer (convolutional) neural networks.

* Motivates *stochastic gradient descent* with mini-batches
* Shows benefits of mean subtraction, normalization, and decorrelation
* Shows drawbacks of sigmoid activation function and motivates *hyperbolic tangent* (tanh)
* Proposes weight initialization scheme (*LeCun initialization*)
* Motivates use of *adaptive optimization* techniques and *momentum*

## Pre-AlexNet

##### [Greedy Layer-Wise Training of Deep Networks](https://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf) (Bengio et al 2006)

Introduces *unsupervised pre-training* and shows significant convergence improvements and generalization performance.

##### [Understanding the Difficulty of Training Deep Feedforward Neural Networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) (Glorot et al 2010)

Shows why training deep neural networks in deep networks is difficult and gives pointers for improvements.

* Gradient propagation study with sigmoid, tanh, and softsign
* New initialization scheme for these activations (*Xavier initialization*)
* Motivates the *cross entropy* loss function instead of mean squared error (MSE)

##### [Deep Sparse Rectifier Neural Networks](http://www.jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf) (Glorot et al 2011)

Shows the advantages of rectified activation functions (*ReLU*) for convergence speed.

##### [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) (Duchi et al 2011)

Introduces *Adagrad*, an adaptive optimization technique.

##### [Practical Recommendations for Gradient-based Training of Deep Architectures](http://arxiv.org/pdf/1206.5533v2.pdf) (Bengio et al 2012)

Practical recommendations for setting hyperparameters such as the learning rate, learning rate decay, batch size, momentum, weight decay, and nonlinearity.

## Post-AlexNet

##### [ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) (Krizhevsky et al 2012)

Breakthrough paper that popularized convolutional neural networks (namely *AlexNet*) and made the following contributions.

* The use of *Local Response Normalization*
* Extensive use of regularizers such as *data augmentation* and *dropout*

##### [Improving Neural Networks by Preventing Co-adaptation of Feature Detectors](http://arxiv.org/pdf/1207.0580.pdf) (Hinton et al 2012)

Paper decribing the dropout regularization technique in detail.

##### [Adadelta: An Adaptive Learning Rate Method](http://arxiv.org/pdf/1212.5701v1.pdf) (Zeiler et al 2012)

Introduces *Adadelta*, an improved version of the Adagrad adaptive optimization technique.

##### [Maxout Networks](http://jmlr.org/proceedings/papers/v28/goodfellow13.pdf) (Goodfellow et al 2013)

Introduces the *maxout neuron*, a companion to dropout, that is able to approximate activation functions such as ReLU and the absolute value.

##### [Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks](http://arxiv.org/pdf/1312.6120v3.pdf) (Saxe et al 2013)

Theoretical analysis of the dynamics in deep neural networks.

##### [On the Inportance of Initialization and Momentum in Deep Learning](http://www.cs.utoronto.ca/~ilya/pubs/2013/1051_2.pdf) (Sutskever et al 2013)

Shows why careful weight initialization and (Nesterov) momentum accelerated SGD are cruciual for training deep neural networks.

## Batch Normalization

##### [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/pdf/1502.03167v3.pdf) (Ioffe et al 2015)

Introduces *batch normalization*.
