# Introduction
In this file we set up a  neural net with an input layer and an output layer.
This neural net will be very basic and we will utilize a linear activation unit and a mean squared error cost
function to create a two-layer neural network for analyzing the titanic dataset from Kaggle. In later tutorials on
this data set we will implement more advanced features.

# Table of contents

* [Model Representation](#model-representation)
* [Neuron Activation and Interaction](#neuron-activation-and-interaction)
* [Cost Function and Gradient Descent](#cost-function-and-gradient-descent)
* [Learning Rate](#learning-rate)

# Model Representation

In a neural network, we typically organize our neurons into layers. The more layers we have, the more complex our
model representation becomes mathematically. However, no matter what, in a neural network we will need a minimum of
two layers; one for input, and one for output. Any layers between are known colloquially as "hidden" layers.

Cool, so starting with the input layer, what does a typical input layer look like? Well it depends on our input.
If we took only one input example we would typically see a vector of features. Features in this case are just
representations of your data that can be used in numeric computation. This means that for some data (like the
titanic data set) we may need to do some transformation if it is not already in numeric format. E.g. the male/female
classification given to us by kaggle can be represented as 0 for male and 1 for female. The information conveyed is
the same, but in the latter representation we can use maths on it! For these input vectors, we can imagine it as
each vector component being a neuron itself in the network. Thus, if we have an input of 6 features, our network's
first layer looks like:

    O
    O
    O
    O
    O
    O

Now let's move to the output layer. The output layer can be thought of in two regards: classification and regression.
an example of a binary classification output could be numbers (like in the classic MNIST tutorials). We can have a
neuron for each single digit number. This would connect directly to our input layer very easily. For the following,
assume that every neuron in the previous layer connects to every neuron in the next.


    O   O
    O   O
    O   O
    O   O
    O   O
    O   O
        O
        O
        O
        O

We could also represent a classification of the numbers based on their binary representation (we need 4 neurons to
represent the 4 bits needed for 10 digits)

    O
    O   O
    O   O
    O   O
    O   O
    O

There are some classification neural nets that output only one value using a transformation called softmax. I won't
go into details, but this should be mentioned.

If we instead were performing regression we could think of the output layer as having a neuron for each predicted
value we want. Here is an example in which we predict 2 values

    O
    O
    O   O
    O   O
    O
    O

We could also just predict one value

    O
    O
    O
    O   O
    O
    O

However these representations are pretty boring with nothing between the two layers. Let's spice things up and add
a hidden layer with 5 units.

    O
    O   O
    O   O
    O   O   O
    O   O
    O   O

What does that do? Well it lets us analyze our data using non-linear boundaries. In layman's speak, if we had data
plotted on a graph we could only separate it using a straight line if using only the input and output layers, while
the addition of the hidden layer lets us separate using a curved continuous boundary. What happens then if we add
another?

Intuitively it would seem that we could get better accuracy this way. Indeed, some data is analyzed very well with
this extra layer, however for many data sets, analysis is best left to a single layer because as you abstract
the model into higher dimensions, the possible boundaries increases and you risk getting non-optimal behavior due
to its ability to find discontinuous solution sets. This is math speak for "using a single hidden layer is usually
good enough."

# Neuron Activation and Interaction

Another part of neural network design is the activation function that drives it. The activation function of a neuron
is one factor in how a neuron handles its input and changing this can drastically change your network's behavior.
The purpose of an activation function is to take an input, and transform it into another output which will be used
the next network layer.

The two major activation functions you see when learning about neural nets are the linear activation function and
the logistic activation function:

    linear activation: f(x) = m * x + b
    logistic activation: f(x) = (1 + e ^ (-(x * w))) ^ -1

These activations have very different behavior. The linear activation function takes an input in the range of
[-inf, inf] and transforms it to an output in the range [-inf, inf]. The logistic activation function on the other
hand takes the same input from [-inf, inf] and transforms it into an output in the range (0, 1).

There are pretty interesting implications in which you use. For instance, if we were to construct a neural network
with one hidden layer using the linear function, we are essentially just propagating information without any
non-linear transformation which turns out to give the same performance of a neural network with just the input and
output layer. In other words, we still aren't getting non-linear separation on our data which is usually our goal
in using neural nets. For this reason, you won't see any multi-layer neural networks that use a linear activation function for their hidden layers.

Note that normally a linear function has the form `w * x + b` for scalar `w` and `b`. However, because the neural net
will be utilizing vectors for w and b, we can get clever and rewrite this to x * w and add a column to both x and w
which we will denote the bias column.

e.g. for a single neural layer:

    we have a vector of weights:

              [2.1,
               3.1,
               2.2,
               0.2]

    and a matrix of features:

              [[0,   1,   2,   3]]
              [[0,   1,   2,   3]]
              [[0,   1,   2,   3]]
              [[0,   1,   2,   3]]
              [[0,   1,   2,   3]]

    and some scalar bias value b (let's say 100 so it stands out).

since the bias for each input x is a scalar, we can pretend that we just have another neuron
with an unvarying x input (the value will be 1) in the layer. This is what we are doing by adding a column.
If this sounds confusing, don't worry, seeing the process should clear things up.

first we take our weights vector and add our bias value to the beginning. We now have:

    w = [100,
         2.1,
         3.1,
         2.2,
         0.2]

Okay now obviously this is gonna cause an error as is because we have a dimension mismatch. We need to resolve this
to make this work. I said we were going to add a column of ones to the x input. In case it is not obvious, we use
ones because this is the identity value in multiplication. If we used zeros, then intuitively it's the same as if
we just dropped the bias value (we'll see later on that this has its uses too!). By adding our column of ones,
the dimensions for matrix multiplication is correct and we get:

    w: [100,
        2.1,
        3.1,
        2.2,
        0.2]

    x: [[1,   0,   1,   2,   3]]
       [[1,   0,   1,   2,   3]]
       [[1,   0,   1,   2,   3]]
       [[1,   0,   1,   2,   3]]
       [[1,   0,   1,   2,   3]]

Note that we could use any other scalar value for padding the feature matrix and it would work, but it is much
easier to just use ones and let the neural net figure out how to adjust the weight representation of the bias. For the
rest of this tutorial, we will denote the linear activation `w * x + b` as `a(x)`.

# Cost Function and Gradient Descent

But wait! We need to have a proper way to adjust these weights in an empirical manner. We first need a way to
describe what a good or bad choice of weights is. We want some kind of function based on our activation function
that imposes a penalty for weights that give us incorrect values. In typical regression we see this in the form of
a cost function that is based on the error of the outputs. When we train a supervised neural net we have both
the data and the result we want, so we can take the output from our activation function and compare them. I'll denote
the true result as `y`. The farther off `a(x)` is from the actual value `y`, the higher error we have.

There are many ways to measure error, however. Sparing the rest, for now we will look at a method known as the mean
squared error (MSE). The idea behind MSE is that for each input, you square the difference between `a(x)`
and `y` (squaring is just to ensure it is a positive value to remove ), then we take the mean of these
squared differences.

Thus, the cost against which we adjust our parameters will be in the form:

      SUM(a(x) - y) ^ 2)/2N

      w = weights
      x = features
      y = result
      N = number of inputs

(Why have we divided by 2N and not just N? This is because later on we will look at how to
  use gradient descent to update our parameters for w. When we do this, this 1/2 will go away and leave a cleaner
  equation for us. It doesn't affect the goal we have regardless of whether it is /2N or /N
  or even /100N.)

Our neural layer's cost (error) for a set of data can be represented by a graph, but what would it look like?

It can be tempting to say it would be linear because of our choice in activation function, but what the cost
function is doing isn't normal use of the linear function. Instead of assuming that `x` is our variable and `w` is
our constant; we do the reverse!

We could plot an example based on the previous example weights and bias, but for each neuron in your neural layer,
this adds another axis against which we plot the result of our cost function. Obviously we can't easily represent
such a high-dimensional plot. However, lets just work with only a single weight and bias unit for this part. The
cost function now goes back to an expanded form. This is surely a confusing jump back and forth, but bear with me:

    Cost = SUM((((w * x) + b) - y) ^ 2)/2N

We can fairly easily represent this function using a 3D graph. Wikipedia gives the perfect image to visualize this:
![](https://upload.wikimedia.org/wikipedia/commons/6/6d/Error_surface_of_a_linear_neuron_with_two_input_weights.png)

I urge you to ignore the numbers and just look at the shape.

The stereotypical next step is to tell you to imagine that your cost is represented by a ball at the top of that
shape's edges (high cost). We want to get the ball into the deepest part of the bowl (minimum cost) and we want to do
so in a generalized way. Lucky for us, this kind of problem is easily solvable using differential calculus.

Going back to the ball idea, we can pretend that the ball's momentum toward the deep part of the bowl can be represented
by another set of equations related to the variables `w` and `b`. We will then use this instantaneous 'momentum'
to update the ball's position manually. This certainly sounds like a job for derivatives for instantaneous rate of
change :).

To get the equation for gradient descent on both variables, we need the derivative of the cost function with respect
to each variable. We do this via partial differentiation. If you aren't familiar with partial differentiation, don't
worry, it's pretty easy.

For the first variable, `w`, we pretend `b` is a constant when differentiating (I will assume you know how to
differentiate using the chain rule):

    partial_derivative_w = SUM(x * (((x * w) + b) - y))/N

Because we divided the cost function by a factor 2 earlier on we get a nice clean equation when we derivate :).
Now lets do the bias `b` with `w` as a constant:

    partial_derivative_b = SUM(((x * w) + b) - y)/N

Before we update our parameters, lets look at the intuition behind these partial derivatives. When we have a high
positive error with respect to a parameter (`y << ((x * w) + b)`) it means that one or both of our parameters is
too high. In this case we want to use the instantaneous rate of change (ROC) as given by the partial derivative to lower
the value of the appropriate parameters. If we have a large negative error (`y >> ((x * w) + b)`) then we want to
use the ROC given by the partial derivatives to increase the appropriate parameter. If we have zero error, then
our derivatives will be zero and we do nothing. We can get this behavior by using a single algorithm:

    w = w - partial_derivative_w
    b = b - partial_derivative_b

By subtracting the partial derivative, we will ensure that we always increase a parameter if it is too low and
vice versa. For zero values, the assignments reduce to w = w and b = b.

    Note: Because the partial_derivative_* equations depend on the `w` and `b` values,
    the code would actually become:

        pd_w = SUM(x * (((x * w) + b) - y))/N
        pd_b = SUM(((x * w) + b) - y)/N

        w = w - pd_w
        b = b - pd_b

    Always perform a simultaneous update when doing gradient descent. If you update by
    calculating inline for each parameter you will skew each subsequent parameter's
    update which will prevent your parameters from converging at minimum-error values.

# Learning rate

Now that we have that settled, we need to address a hypothetical issue (one that turns out to be a real practical
problem). Suppose we run our gradient descent algorithm and we have a large adjustment we need to make to our
parameters based on our partial derivatives. When we make our adjustments and reiterate through, we find that we
actually overshot the minimum by a large amount, and again we need to make another large adjustment, but in the
opposite direction this time! We keep doing this and our algorithm fails to cause the parameters to converge on the
best values. This is known as overshooting and it happens in real neural network applications!

We can correct this behavior by introducing a scaling factor, `alpha`, into our update code. This `alpha` will be a
scalar constant which will scale the partial derivatives so that we update our parameters by a smaller amount to
prevent overshooting. Our algorithm then becomes:

    w = w - alpha*partial_derivative_w
    b = b - alpha*partial_derivative_b

So what values do we pick for our alpha? Well, that depends and you'll need to do a bit of personal debugging to
determine a good alpha value. Sometimes a value of 1 is good, but other times you notice overshooting and will
adjust to, say, 0.001 or 1e-3. However, this can cause cause "undershooting" which at worst may find a local minimum
(remember, while our example for now is that simple bowl shape, your neural net will be represented by a multitude
of valleys and peaks), but not a global one, and at best will cause your net to learn slowly. The short answer is
that you'll need to monitor what your neural net is doing to determine how to adjust your alpha value.

Note that in modern machine learning frameworks (as we'll see later) usually there are optimizers that handle this
choice of alpha for you and can even do dynamic adjustments to alpha during training!

Now that we know how things work in a single neuron, we need to switch back into the mindset of neural networks!
In the previous examples for the neural network, we had our weights and inputs organized so that the bias term was
included in the weights vector and a column of ones was introduced into the input.

This means that our partial differential equations for bias and weight can also be combined like the linear
activation function as such:

    partial_derivative_w = SUM(x * ((x * w) - y))/N
    w = w - alpha * partial_derivative_w

This certainly cleans things up. However a thought occurs... in a multi-layer neural network, our activations of
each subsequent layer, from the first hidden layer to the output layer, depend on each other. If I have a neural
network with 5 input features,

BACKPROP

This is pretty much all you need to know for writing the neuron activation and gradient descent. Let's get to the
code then! I will comment many parts of the code for convenience :)
