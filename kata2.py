import os
import shutil

import numpy as np
from neuraxle.base import BaseStep, NonFittableMixin, Identity
from neuraxle.hyperparams.distributions import Choice, Boolean
from neuraxle.hyperparams.distributions import RandInt, LogUniform
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import AutoML, InMemoryHyperparamsRepository, validation_splitter, \
    RandomSearchHyperparameterSelectionStrategy
from neuraxle.metaopt.callbacks import ScoringCallback
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import TrainOnlyWrapper, ChooseOneStepOf
from neuraxle.steps.numpy import NumpyConcatenateInnerFeatures, NumpyShapePrinter, NumpyFlattenDatum
from neuraxle.steps.output_handlers import OutputTransformerWrapper
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.union import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from data_loading import load_all_data


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

    decision_tree_classifier = SKLearnWrapper(
        DecisionTreeClassifier(), HyperparameterSpace({
            'criterion': Choice(['gini', 'entropy']), 'splitter': Choice(['best', 'random']),
            'min_samples_leaf': RandInt(2, 5), 'min_samples_split': RandInt(1, 3), }))

    extra_tree_classifier = SKLearnWrapper(
        ExtraTreeClassifier(), HyperparameterSpace({
            'criterion': Choice(['gini', 'entropy']), 'splitter': Choice(['best', 'random']),
            'min_samples_leaf': RandInt(2, 5), 'min_samples_split': RandInt(1, 3), }))

    ridge_classifier = Pipeline([
        OutputTransformerWrapper(NumpyRavel()),
        SKLearnWrapper(RidgeClassifier(), HyperparameterSpace({
            'alpha': Choice([(0.0, 1.0, 10.0), (0.0, 10.0, 100.0)]), 'fit_intercept': Boolean(), 'normalize': Boolean()
        }))
    ]).set_name('RidgeClassifier')

    logistic_regression = Pipeline([
        OutputTransformerWrapper(NumpyRavel()),
        SKLearnWrapper(LogisticRegression(), HyperparameterSpace({
            'C': LogUniform(0.01, 10.0), 'fit_intercept': Boolean(), 'dual': Boolean(),
            'penalty': Choice(['l1', 'l2']), 'max_iter': RandInt(20, 200)
        }))
    ]).set_name('LogisticRegression')

    random_forest_classifier = Pipeline([
        OutputTransformerWrapper(NumpyRavel()),
        SKLearnWrapper(RandomForestClassifier(), HyperparameterSpace({
            'n_estimators': RandInt(50, 600), 'criterion': Choice(['gini', 'entropy']),
            'min_samples_leaf': RandInt(2, 5), 'min_samples_split': RandInt(1, 3),
            'bootstrap': Boolean()
        }))
    ]).set_name('RandomForestClassifier')

    pipeline = Pipeline([
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
        # TODO in kata 2, optional: Add some feature selection right here for the motivated ones:
        #      https://scikit-learn.org/stable/modules/feature_selection.html
        # TODO in kata 2, optional: Add normalization right here (if using other classifiers)
        #      https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
        ChooseOneStepOf([
            decision_tree_classifier,
            extra_tree_classifier,
            ridge_classifier,
            logistic_regression,
            random_forest_classifier
        ]),
        TrainOnlyWrapper(NumpyShapePrinter(custom_message="Shape at output after classification")),
        # Shape: [batch_size]
        Identity()
    ])

    auto_ml = AutoML(
        pipeline=pipeline,
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        validation_split_function=validation_splitter(test_size=0.20),
        scoring_callback=ScoringCallback(accuracy_score, higher_score_is_better=False),
        n_trials=7,
        epochs=1,
        hyperparams_repository=InMemoryHyperparamsRepository(cache_folder=cache_folder),
        refit_trial=True,
    )

    # Do AutoML and get best model refitted:
    X_train, y_train, X_test, y_test = load_all_data()

    auto_ml = auto_ml.fit(X_train, y_train)
    best_pipeline = auto_ml.get_best_model()

    # Predict on test data and score:
    y_pred = best_pipeline.predict(X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Test accuracy score:", accuracy)
    assert accuracy > 0.85
    # It's getting good on this dataset if you're over 91%. The current code is able to do this.
    # Getting 94% would be very hard.


if __name__ == '__main__':
    main()
