"""
    In this file we set up a  neural net with an input layer, hidden layer, and an output layer.
    This neural net will be very basic and we will utilize a linear activation unit and a mean squared error cost
    function.

    This means that for our neural net, the cost against which we adjust our parameters will be in the form:

        MEAN SQUARED ERROR
            SUM((x * w) - y) ^ 2)/2N

            w = weights
            x = features
            y = result
            N = number of inputs

    (First let me address why we have it divided by 2N and not just N. This is because later on we will look at how to
    use gradient descent to update our parameters for w. When we do this, this 1/2 will go away and leave a cleaner
    equation for us. It doesn't affect the goal we have regardless of whether it is /2N or /N
    or even /100N.)

    Note that normally a linear function has the form w * x + b for scalar w and b. However, because the neural net
    will be utilizing vectors for w and b, we can get clever and rewrite this to x * w and add a column to both x and w
    which we will denote the bias column.

    e.g. for a single neural layer:
        we have a vector of weights: [2.1,
                                      3.1,
                                      2.2,
                                      0.2]

            and a matrix of features: [[0,   1,   2,   3]]
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
    easier to just use ones and let the neural net figure out how to adjust the weight representation of the bias.

    But wait! We need to have a proper way to adjust these weights in an empirical manner. Before we get to this,
    we can realize our neural layer's cost (error) for a set of data can be represented by a graph.
    What would it look like?

    It can be tempting to say it would be linear because of our choice in activation function, but what the cost
    function is doing isn't normal use of the linear function. Instead of assuming that x is our variable and w and b
    are our constants; we do the reverse!

    We could plot an example based on the previous example weights and bias, but for each neuron in your neural layer,
    this adds another axis against which we plot the result of our cost function. Obviously we can't easily represent
    such a high-dimensional plot. However, lets just work with only a single weight and bias unit for this part. The
    cost function now becomes:

        SUM((((w * x) + b) - y) ^ 2)/2N

    We can fairly easily represent this function using a 3D graph. I'm going to be shameless and give you a link to use
    for this: https://upload.wikimedia.org/wikipedia/commons/6/6d/Error_surface_of_a_linear_neuron_with_two_input_weights.png
    I urge you to ignore the numbers and just look at the shape.

    The stereotypical next step is to tell you to imagine that your cost is represented by a ball at the top of that
    shape's edges (high cost). We want to get the ball into the deepest part of the bowl and we want to do so in a
    generalized way. Lucky for us, this kind of problem is easily solvable using differential calculus.

    Going back to the ball, we can pretend that the ball's momentum toward the deep part of the bowl can be represented
    by another set of equations related to the variables w and b. We will then use this instantaneous momentum to update
    the ball's position manually. This certainly sounds like a job for derivatives for instantaneous rate of change :).

    To get the equation for gradient descent on both variables, we need the derivative of the cost function with respect
    to each variable. If you aren't familiar with partial differentiation, don't worry, it's pretty easy.

    For the first variable, w, we pretend b is a constant when differentiating (I will assume you know how to
    differentiate using the chain rule):

        partial_derivative_w = SUM(x * (((x * w) + b) - y))/N

    Because we divided the cost function by 2 we get a nice clean equation when we derivate :). Now the bias b:

        partial_derivative_b = SUM(((x * w) + b) - y)/N

    Before we update our parameters, lets look at the intuition behind these partial derivatives. When we have a high
    positive error with respect to a parameter (y << ((x * w) + b)) it means that one or both of our parameters is
    too high. In this case we want to use the instantaneous rate of change (ROC) as given by the partial derivative to lower
    the value of the appropriate parameters. If we have a large negative error (y >> ((x * w) + b)) then we want to
    use the ROC given by the partial derivatives to increase the appropriate parameter. If we have zero error, then
    our derivatives will be zero and we do nothing. We can get this behavior by using a single algorithm:

        w = w - partial_derivative_w
        b = b - partial_derivative_b

    By subtracting the partial derivative, we will ensure that we always increase a parameter if it is too low and
    vice versa. For zero values, the assignments reduce to w = w and b = b.

        Note: Because the partial_derivative_* equations depend on the w and b, the code would actually become

            pd_w = SUM(x * (((x * w) + b) - y))/N
            pd_b = SUM(((x * w) + b) - y)/N

            w = w - pd_w
            b = b - pd_b

        always perform a simultaneous update when doing gradient descent. If you update by calculating inline for each
        parameter you will skew each subsequent parameter's update which will prevent your parameters from converging
        at minimum-error values.

    Now that we have that settled, we need to address a hypothetical issue (one that turns out to be a real practical
    problem). Suppose we run our gradient descent algorithm and we have a large adjustment we need to make to our
    parameters. When we make our adjustments and reiterate through, we find that we actually overshot the minimum by a
    large amount, and again we need to make another large adjustment, but in the opposite direction this time! We keep
    doing this and our algorithm fails to cause the parameters to converge on the best values. This is known as
    overshooting and it happens in real neural network applications!

    We can correct this behavior by introducing a scaling factor, alpha, into our update code. This alpha will be a
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

    

"""
