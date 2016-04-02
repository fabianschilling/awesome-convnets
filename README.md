# Awesome Convolutional Neural Networks

A list of papers I used for my thesis about convolutional neural networks with a focus on batch normalization.

## Early work

##### [Learning Representations by Backpropagating Errors](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) (Rumelhart et al 1986)

Invention of the backpropagation algorithm.

##### [Backpropagation Applied to Handwritten Zip Code Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf) (LeCun et al 1989)

Backpropagation applied to character recognition with weight sharing.

##### [Gradient-based Learning Applied to Document Recogintion](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (LeCun et al 1998)

Defines convolutional neural networks.

##### [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (LeCun et al 1998)

Gives many practical recommendations for traning multi-layer (convolutional) neural networks.

* Motivates stochastic gradient descent with mini-batches
* Shows benefits of mean subtraction, normalization, and decorrelation
* Shows drawbacks of sigmoid activation function and motivates hyperbolic tangent (tanh)
* Proposes weight initialization scheme (LeCun initialization)
* Motivates use of adaptive optimization techniques and momentum

## Pre-AlexNet

##### [Understanding the Difficulty of Training Deep Feedforward Neural Networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) (Glorot et al 2010)

Shows why training deep neural networks in deep networks is difficult and gives pointers for improvements.

* Gradient propagation study with sigmoid, tanh, and softsign
* New initialization scheme for these activations (Xavier initialization)
* Motivates the cross entropy loss function instead of mean squared error (MSE)

##### [Deep Sparse Rectifier Neural Networks](http://www.jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf) (Glorot et al 2011)

Shows the advantages of rectified activation functions for convergence speed.

##### [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) (Duchi et al 2011)

Introduces Adagrad, an adaptive optimization technique.

##### [Practical Recommendations for Gradient-based Training of Deep Architectures](http://arxiv.org/pdf/1206.5533v2.pdf) (Bengio et al 2012)

Practical recommendations for setting hyperparameters such as the learning rate, learning rate decay, batch size, momentum, weight decay, and nonlinearity.
