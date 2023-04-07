# Kagge-Skin-Cancer-Malignant-vs.-Benign-with-CNN



Hi,
İmport ad Load
Firstly if you want to review this data set you click the link https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign
We have the data set that include two classes which benign images and malignant images.And we make a classification on this data set.
We import required libraries and load our data set.
Processing
And we must be 'rgb' image to use in convolutional network algorithms and we can make prosess it while implement the DataGenerator.We describe required function with 'scale, width_shift_range,height_shift_range,zoom_range' etc.
Actually these functions are hyperparametes for us, to use our aim is be find the best score with these operations

Model

And than we may create a model to classify, we may add 'filters, kernel_size, input_shape, activation function' ets operations, In addition we have to implement max_polling and flatten and dropout operation to our model.Finally compile this model with that I describe kind of the 'optimizers, loss functions and metrics 'ets.
We have to use epoch numbers and batch size etc while we fit the our model with values or functions that we adding and describe.We observe lost and accuracy end of the fit operation.

And we test this model with our test images





