import os
import shutil

import numpy as np
from neuraxle.base import BaseStep, NonFittableMixin
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import AutoML, InMemoryHyperparamsRepository, validation_splitter, \
    RandomSearchHyperparameterSelectionStrategy
from neuraxle.metaopt.callbacks import ScoringCallback, MetricCallback
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.numpy import NumpyConcatenateInnerFeatures, NumpyShapePrinter, NumpyFlattenDatum
from neuraxle.steps.output_handlers import OutputTransformerWrapper
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.union import FeatureUnion
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier, LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from auto_ml_steps import ChooseOneStepOf, ChooseOneOrManyStepsOf
from data_loading import load_all_data_without_split


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


class NumpyAbs(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        """
        Will featurize data with a max.

        :param data_inputs: 3D time series of shape [batch_size, time_steps, sensors]
        :return: featurized time series of shape [batch_size, features]
        """
        return np.abs(data_inputs)


class NumpyMean(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        """
        Will featurize data with a mean.

        :param data_inputs: 3D time series of shape [batch_size, time_steps, sensors]
        :return: featurized time series of shape [batch_size, features]
        """
        return np.mean(data_inputs, axis=-2)

class NumpyRavel(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        if data_inputs is not None:
            data_inputs = data_inputs if isinstance(data_inputs, np.ndarray) else np.array(data_inputs)
            return data_inputs.ravel()
        return data_inputs


class NumpyMedian(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        """
        Will featurize data with a median.

        :param data_inputs: 3D time series of shape [batch_size, time_steps, sensors]
        :return: featurized time series of shape [batch_size, features]
        """
        return np.median(data_inputs, axis=-2)


class NumpyMin(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        """
        Will featurize data with a min.

        :param data_inputs: 3D time series of shape [batch_size, time_steps, sensors]
        :return: featurized time series of shape [batch_size, features]
        """
        return np.min(data_inputs, axis=-2)


class NumpyMax(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        """
        Will featurize data with a max.

        :param data_inputs: 3D time series of shape [batch_size, time_steps, sensors]
        :return: featurized time series of shape [batch_size, features]
        """
        return np.max(data_inputs, axis=-2)


def main():
    cache_folder = 'cache'
    if os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)
    os.makedirs(cache_folder, exist_ok=True)

    auto_ml = AutoML(
        pipeline=Pipeline([
            TrainOnlyWrapper(NumpyShapePrinter(custom_message="Input shape before feature union")),
            FeatureUnion([
                Pipeline([
                    NumpyFFT(),
                    NumpyAbs(),
                    FeatureUnion([
                        NumpyFlattenDatum(),  # Reshape from 3D to flat 2D: flattening data except on batch size
                        FFTPeakBinWithValue()  # Extract 2D features from the 3D FFT bins
                    ], joiner=NumpyConcatenateInnerFeatures())
                ]),
                NumpyMean(),
                NumpyMedian(),
                NumpyMin(),
                NumpyMax()
            ], joiner=NumpyConcatenateInnerFeatures()),
            TrainOnlyWrapper(NumpyShapePrinter(custom_message="Shape after feature union, before classification")),
            # Shape: [batch_size, remade_features]
            # TODO in kata 2: Add some feature selection right here for the motivated ones:
            #      https://scikit-learn.org/stable/modules/feature_selection.html
            # TODO in kata 2: Add normalization right here (if using other classifiers)
            #      https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
            ChooseOneStepOf([
                SKLearnWrapper(DecisionTreeClassifier(), HyperparameterSpace({ })),
                SKLearnWrapper(ExtraTreeClassifier(), HyperparameterSpace({ })),
                Pipeline([OutputTransformerWrapper(NumpyRavel()), SKLearnWrapper(RidgeClassifier(), HyperparameterSpace({}))]).set_name('RidgeClassifier'),
                Pipeline([OutputTransformerWrapper(NumpyRavel()), SKLearnWrapper(RidgeClassifierCV(), HyperparameterSpace({}))]).set_name('RidgeClassifierCV'),
                Pipeline([OutputTransformerWrapper(NumpyRavel()), SKLearnWrapper(KNeighborsClassifier(), HyperparameterSpace({}))]).set_name('kNeighborsClassifier'),
                Pipeline([OutputTransformerWrapper(NumpyRavel()), SKLearnWrapper(LogisticRegression(), HyperparameterSpace({}))]).set_name('LogisticRegression'),
                Pipeline([OutputTransformerWrapper(NumpyRavel()), SKLearnWrapper(LogisticRegressionCV(), HyperparameterSpace({}))]).set_name('LogisticRegressionCv'),
                Pipeline([OutputTransformerWrapper(NumpyRavel()), SKLearnWrapper(RandomForestClassifier(), HyperparameterSpace({}))]).set_name('RandomForestClassifier'),
            ]),
            # TODO in kata 2: Try other classifiers different than the DecisionTreeClassifier just above:
            #      https://scikit-learn.org/stable/modules/multiclass.html
            TrainOnlyWrapper(NumpyShapePrinter(custom_message="Shape at output after classification")),
            # Shape: [batch_size]
        ]),
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        validation_split_function=validation_splitter(0.20),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False),
            MetricCallback('accuracy', metric_function=accuracy_score, higher_score_is_better=True)
        ],
        n_trials=3,
        refit_trial=True,
        epochs=1,
        hyperparams_repository=InMemoryHyperparamsRepository(cache_folder=cache_folder),
        print_metrics=True
    )

    # When
    data_inputs, expected_outputs = load_all_data_without_split()
    auto_ml = auto_ml.fit(data_inputs, expected_outputs)

    # Then
    p = auto_ml.get_best_model()

    _, _, validation_data_inputs, validation_expected_outputs = validation_splitter(0.20)(data_inputs, expected_outputs)
    y_pred = p.predict(validation_data_inputs)

    accuracy = accuracy_score(validation_expected_outputs, y_pred)
    print("Test accuracy score:", accuracy)
    assert accuracy > 0.7


if __name__ == '__main__':
    main()