# Kata: Clean Machine Learning From Dirty Code

You can clone this project and launch jupyter-notebook, or use the files in Google Colab here: 

- https://drive.google.com/drive/u/0/folders/12uzcNKU7n0EUyFzgitSt1wSaSvV4qJbs

You may want to do `File > Save a copy in Drive...` in Colab to edit the file.

___

# Kata 1: Refactor Dirty ML Code into Pipeline

Let's convert dirty machine learning code into clean code using a [Pipeline](https://stackoverflow.com/a/60303302/2476920) - which is the [Pipe and Filter Design Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/pipes-and-filters) for Machine Learning. 

At first you may still wonder *why* using this Design Patterns is good. You'll realize just how good it is in the 2nd [Clean Machine Learning Kata](https://github.com/Neuraxio/Kata-Clean-Machine-Learning-From-Dirty-Code) when you'll do AutoML. Pipelines will give you the ability to easily manage the hyperparameters and the hyperparameter space, on a per-step basis. You'll also have the good code structure for training, saving, reloading, and deploying using any library you want without hitting a wall when it'll come to serializing your whole trained pipeline for deploying in prod.


## The Dataset

It'll be downloaded automatically for you in the code below. 

We're using a Human Activity Recognition (HAR) dataset captured using smartphones. The [dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) can be found on the UCI Machine Learning Repository. 

### The task

Classify the type of movement amongst six categories from the phones' sensor data:
- WALKING,
- WALKING_UPSTAIRS,
- WALKING_DOWNSTAIRS,
- SITTING,
- STANDING,
- LAYING.

### Video dataset overview

Follow this link to see a video of the 6 activities recorded in the experiment with one of the participants:

<p align="center">
  <a href="http://www.youtube.com/watch?feature=player_embedded&v=XOEN9W05_4A
" target="_blank"><img src="http://img.youtube.com/vi/XOEN9W05_4A/0.jpg" 
alt="Video of the experiment" width="400" height="300" border="10" /></a>
  <a href="https://youtu.be/XOEN9W05_4A"><center>[Watch video]</center></a>
</p>

### Details about the input data

The dataset's description goes like this:

> The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. 

Reference: 
> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

That said, I will use the almost raw data: only the gravity effect has been filtered out of the accelerometer  as a preprocessing step for another 3D feature as an input to help learning. If you'd ever want to extract the gravity by yourself, you could use the following [Butterworth Low-Pass Filter (LPF)](https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform) and edit it to have the right cutoff frequency of 0.3 Hz which is a good frequency for activity recognition from body sensors.

Here is how the 3D data cube looks like. So we'll have a train and a test data cube, and might create validation data cubes as well: 

![](time-series-data.jpg)

So we have 3D data of shape `[batch_size, time_steps, features]`. If this and the above is still unclear to you, you may want to [learn more on the 3D shape of time series data](https://www.quora.com/What-do-samples-features-time-steps-mean-in-LSTM/answer/Guillaume-Chevalier-2).

## Loading the Dataset


```python
import urllib
import os

def download_import(filename):
    with open(filename, "wb") as f:
        # Downloading like that is needed because of Colab operating from a Google Drive folder that is only "shared with you".
        url = 'https://raw.githubusercontent.com/Neuraxio/Kata-Clean-Machine-Learning-From-Dirty-Code/master/{}'.format(filename)
        f.write(urllib.request.urlopen(url).read())

try:
    import google.colab
    download_import("data_loading.py")
    !mkdir data;
    download_import("data/download_dataset.py")
    print("Downloaded .py files: dataset loaders.")
except:
    print("No dynamic .py file download needed: not in a Colab.")

DATA_PATH = "data/"
!pwd && ls
os.chdir(DATA_PATH)
!pwd && ls
!python download_dataset.py
!pwd && ls
os.chdir("..")
!pwd && ls
DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
print("\n" + "Dataset is now located at: " + DATASET_PATH)
```


```python
# install neuraxle if needed:
try:
    import neuraxle
    assert neuraxle.__version__ == '0.3.4'
except:
    !pip install neuraxle==0.3.4
```


```python
# Finally load dataset!
from data_loading import load_all_data
X_train, y_train, X_test, y_test = load_all_data()
print("Dataset loaded!")
```

## Let's now define and execute our ugly code: 

You don't need to change the functions here just below. We'll rather code this again after in the next section.


```python
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def get_fft_peak_infos(real_fft, time_bins_axis=-2):
    """
    Extract the indices of the bins with maximal amplitude, and the corresponding amplitude values.

    :param fft: real magnitudes of an fft. It could be of shape [N, bins, features].
    :param time_bins_axis: axis of the frequency bins (e.g.: time axis before fft).
    :return: Two arrays without bins. One is an int, the other is a float. Shape: ([N, features], [N, features])
    """
    peak_bin = np.argmax(real_fft, axis=time_bins_axis)
    peak_bin_val = np.max(real_fft, axis=time_bins_axis)
    return peak_bin, peak_bin_val


def fft_magnitudes(data_inputs, time_axis=-2):
    """
    Apply a Fast Fourier Transform operation to analyze frequencies, and return real magnitudes.
    The bins past the half (past the nyquist frequency) are discarded, which result in shorter time series.

    :param data_inputs: ND array of dimension at least 1. For instance, this could be of shape [N, time_axis, features]
    :param time_axis: axis along which the time series evolve
    :return: real magnitudes of the data_inputs. For instance, this could be of shape [N, (time_axis / 2) + 1, features]
             so here, we have `bins = (time_axis / 2) + 1`.
    """
    fft = np.fft.rfft(data_inputs, axis=time_axis)
    real_fft = np.absolute(fft)
    return real_fft


def get_fft_features(x_data):
    """
    Will featurize data with an FFT.

    :param x_data: 3D time series of shape [batch_size, time_steps, sensors]
    :return: featurized time series with FFT of shape [batch_size, features]
    """
    real_fft = fft_magnitudes(x_data)
    flattened_fft = real_fft.reshape(real_fft.shape[0], -1)
    peak_bin, peak_bin_val = get_fft_peak_infos(real_fft)
    return flattened_fft, peak_bin, peak_bin_val


def featurize_data(x_data):
    """
    Will convert 3D time series of shape [batch_size, time_steps, sensors] to shape [batch_size, features]
    to prepare data for machine learning.

    :param x_data: 3D time series of shape [batch_size, time_steps, sensors]
    :return: featurized time series of shape [batch_size, features]
    """
    print("Input shape before feature union:", x_data.shape)

    flattened_fft, peak_bin, peak_bin_val = get_fft_features(x_data)
    mean = np.mean(x_data, axis=-2)
    median = np.median(x_data, axis=-2)
    min = np.min(x_data, axis=-2)
    max = np.max(x_data, axis=-2)

    featurized_data = np.concatenate([
        flattened_fft,
        peak_bin,
        peak_bin_val,
        mean,
        median,
        min,
        max,
    ], axis=-1)

    print("Shape after feature union, before classification:", featurized_data.shape)
    return featurized_data

```

Let's now use the ugly code to do ugly machine learning with it.

Fit: 


```python

# Shape: [batch_size, time_steps, sensor_features]
X_train_featurized = featurize_data(X_train)
# Shape: [batch_size, remade_features]

classifier = DecisionTreeClassifier()
classifier.fit(X_train_featurized, y_train)

```

Predict:


```python

# Shape: [batch_size, time_steps, sensor_features]
X_test_featurized = featurize_data(X_test)
# Shape: [batch_size, remade_features]

y_pred = classifier.predict(X_test_featurized)
print("Shape at output after classification:", y_pred.shape)
# Shape: [batch_size]

```

Eval:


```python

accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
print("Accuracy of ugly pipeline code:", accuracy)

```

## Cleaning Up: Define Pipeline Steps and a Pipeline

The kata is to fill the classes below and to use them properly in the pipeline thereafter. 

There are some missing classes as well that you should define.


```python
from neuraxle.base import BaseStep, NonFittableMixin
from neuraxle.steps.numpy import NumpyConcatenateInnerFeatures, NumpyShapePrinter, NumpyFlattenDatum

class NumpyFFT(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        """
        Featurize time series data with FFT.

        :param data_inputs: time series data of 3D shape: [batch_size, time_steps, sensors_readings]
        :return: featurized data is of 2D shape: [batch_size, n_features]
        """
        transformed_data = np.fft.rfft(data_inputs, axis=-2)
        return transformed_data


class FFTPeakBinWithValue(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        """
        Will compute peak fft bins (int), and their magnitudes' value (float), to concatenate them.

        :param data_inputs: real magnitudes of an fft. It could be of shape [batch_size, bins, features].
        :return: Two arrays without bins concatenated on feature axis. Shape: [batch_size, 2 * features]
        """
        time_bins_axis = -2
        peak_bin = np.argmax(data_inputs, axis=time_bins_axis)
        peak_bin_val = np.max(data_inputs, axis=time_bins_axis)
        
        # Notice that here another FeatureUnion could have been used with a joiner:
        transformed = np.concatenate([peak_bin, peak_bin_val], axis=-1)
        
        return transformed


class NumpyMedian(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        """
        Will featurize data with a median.

        :param data_inputs: 3D time series of shape [batch_size, time_steps, sensors]
        :return: featurized time series of shape [batch_size, features]
        """
        return np.median(data_inputs, axis=-2)


class NumpyMean(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        """
        Will featurize data with a mean.

        :param data_inputs: 3D time series of shape [batch_size, time_steps, sensors]
        :return: featurized time series of shape [batch_size, features]
        """
        raise NotImplementedError("TODO")
        return ...

```

Let's now create the Pipeline with the code:


```python
from neuraxle.base import Identity
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.union import FeatureUnion

pipeline = Pipeline([
    # ToNumpy(),  # Cast type in case it was a list.
    # For debugging, do this print at train-time only:
    TrainOnlyWrapper(NumpyShapePrinter(custom_message="Input shape before feature union")),
    # Shape: [batch_size, time_steps, sensor_features]
    FeatureUnion([
        # TODO in kata 1: Fill the classes in this FeatureUnion here and make them work.
        #      Note that you may comment out some of those feature classes
        #      temporarily and reactivate them one by one.
        Pipeline([
            NumpyFFT(),
            NumpyAbs(),  # do `np.abs` here.
            FeatureUnion([
                NumpyFlattenDatum(),  # Reshape from 3D to flat 2D: flattening data except on batch size
                FFTPeakBinWithValue()  # Extract 2D features from the 3D FFT bins
            ], joiner=NumpyConcatenateInnerFeatures())
        ]),
        NumpyMean(),
        NumpyMedian(),
        NumpyMin(),
        NumpyMax()
    ], joiner=NumpyConcatenateInnerFeatures()),  # The joiner will here join like this: np.concatenate([...], axis=-1)
    # TODO, optional: Add some feature selection right here for the motivated ones:
    #      https://scikit-learn.org/stable/modules/feature_selection.html
    TrainOnlyWrapper(NumpyShapePrinter(custom_message="Shape after feature union, before classification")),
    # Shape: [batch_size, remade_features]
    # TODO: use an `Inherently multiclass` classifier here from:
    #       https://scikit-learn.org/stable/modules/multiclass.html
    YourClassifier(),
    TrainOnlyWrapper(NumpyShapePrinter(custom_message="Shape at output after classification")),
    # Shape: [batch_size]
    Identity()
])

```

## Test Your Code: Make the Tests Pass

The 3rd test is the real deal.


```python
def _test_is_pipeline(pipeline):
    assert isinstance(pipeline, Pipeline)


def _test_has_all_data_preprocessors(pipeline):
    assert "DecisionTreeClassifier" in pipeline
    assert "FeatureUnion" in pipeline
    assert "Pipeline" in pipeline["FeatureUnion"]
    assert "NumpyMean" in pipeline["FeatureUnion"]
    assert "NumpyMedian" in pipeline["FeatureUnion"]
    assert "NumpyMin" in pipeline["FeatureUnion"]
    assert "NumpyMax" in pipeline["FeatureUnion"]


def _test_pipeline_words_and_has_ok_score(pipeline):
    pipeline = pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy score:", accuracy)
    assert accuracy > 0.7


if __name__ == '__main__':
    tests = [_test_is_pipeline, _test_has_all_data_preprocessors, _test_pipeline_words_and_has_ok_score]
    for t in tests:
        try:
            t(pipeline)
            print("==> Test '{}(pipeline)' succeed!".format(t.__name__))
        except Exception as e:
            print("==> Test '{}(pipeline)' failed:".format(t.__name__))
            import traceback
            print(traceback.format_exc())

```

## Good job!

Your code should now be clean after making the tests pass.

## You're ready for the [Kata 2](https://github.com/Neuraxio/Kata-Clean-Machine-Learning-From-Dirty-Code#kata-clean-machine-learning-from-dirty-code).

You should now be ready for the 2nd [Clean Machine Learning Kata](https://github.com/Neuraxio/Kata-Clean-Machine-Learning-From-Dirty-Code#kata-clean-machine-learning-from-dirty-code). Note that the solutions are available in the repository above as well. You may use the links to the Google Colab files to try to solve the Katas. 

___

## Recommended additional readings and learning resources: 

- For more info on clean machine learning, you may want to read [How to Code Neat Machine Learning Pipelines](https://www.neuraxio.com/en/blog/neuraxle/2019/10/26/neat-machine-learning-pipelines.html).
- For reaching higher performances, you could use a [LSTM Recurrent Neural Network](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition) and refactoring it into a neat pipeline as you've created here, now by [using TensorFlow in your ML pipeline](https://github.com/Neuraxio/Neuraxle-TensorFlow).
- You may as well want to request [more training and coaching for your ML or time series processing projects](https://www.neuraxio.com/en/time-series-solution) from us if you need.

