import os
import shutil
from typing import Union

import numpy as np
from neuraxle.base import ForceHandleOnlyMixin, MetaStepMixin, BaseStep, ExecutionContext, NonFittableMixin
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.distributions import Choice, Boolean
from neuraxle.hyperparams.distributions import RandInt, LogUniform
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.metaopt.auto_ml import AutoML, InMemoryHyperparamsRepository, validation_splitter, \
    RandomSearchHyperparameterSelectionStrategy
from neuraxle.metaopt.callbacks import ScoringCallback, MetricCallback
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import SelectNonEmptyDataInputs, CHOICE_HYPERPARAM, OPTIONAL_ENABLED_HYPERPARAM
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.numpy import NumpyConcatenateInnerFeatures, NumpyShapePrinter, NumpyFlattenDatum
from neuraxle.steps.output_handlers import OutputTransformerWrapper
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.union import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier, LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from data_loading import load_all_data


class Optional(ForceHandleOnlyMixin, MetaStepMixin, BaseStep):
    """
    A wrapper to nullify a step : nullify its hyperparams, and also nullify all of his behavior.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            Optional(Identity(), enabled=True)
        ])

    """

    def __init__(self, wrapped: BaseStep, enabled: bool = True, nullified_return_value=None,
                 cache_folder_when_no_handle=None, use_hyperparameter_space=True, nullify_hyperparams=True):
        hyperparameter_space = HyperparameterSpace({
            OPTIONAL_ENABLED_HYPERPARAM: Boolean()
        }) if use_hyperparameter_space else {}

        BaseStep.__init__(
            self,
            hyperparams=HyperparameterSamples({
                OPTIONAL_ENABLED_HYPERPARAM: enabled
            }),
            hyperparams_space=hyperparameter_space
        )
        MetaStepMixin.__init__(self, wrapped)
        ForceHandleOnlyMixin.__init__(self, cache_folder_when_no_handle)

        if nullified_return_value is None:
            nullified_return_value = []
        self.nullified_return_value = nullified_return_value
        self.nullify_hyperparams = nullify_hyperparams

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (
    'BaseStep', DataContainer):
        """
        Nullify wrapped step hyperparams, and don't fit the wrapped step.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        :type: (BaseStep, DataContainer)
        """
        if self.hyperparams[OPTIONAL_ENABLED_HYPERPARAM]:
            self.wrapped = self.wrapped.handle_fit(data_container, context)
            return self

        self._nullify_hyperparams()

        return self

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (
    'BaseStep', DataContainer):
        """
        Nullify wrapped step hyperparams, and don't fit_transform the wrapped step.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        :type: (BaseStep, DataContainer)
        """
        if self.hyperparams[OPTIONAL_ENABLED_HYPERPARAM]:
            self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
            return self, data_container

        self._nullify_hyperparams()

        return self, DataContainer(
            data_inputs=self.nullified_return_value,
            current_ids=data_container.current_ids,
            expected_outputs=self.nullified_return_value
        )

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Nullify wrapped step hyperparams, and don't transform the wrapped step.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        :type: DataContainer
        """
        if self.hyperparams[OPTIONAL_ENABLED_HYPERPARAM]:
            return self.wrapped.handle_transform(data_container, context)

        self._nullify_hyperparams()
        data_container.set_data_inputs(self.nullified_return_value)

        return DataContainer(
            data_inputs=self.nullified_return_value,
            current_ids=data_container.current_ids,
            expected_outputs=self.nullified_return_value
        )

    def _nullify_hyperparams(self):
        """
        Nullify wrapped step hyperparams using hyperparams_space.nullify().
        """
        if not self.nullify_hyperparams:
            return
        hyperparams_space = self.wrapped.get_hyperparams_space()
        self.wrapped.set_hyperparams(hyperparams_space.nullify())


class ChooseOneStepOf(FeatureUnion):
    """
    A pipeline to allow choosing one step using an hyperparameter.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            ChooseOneStepOf([
                ('a', Identity()),
                ('b', Identity())
            ])
        ])
        p.set_hyperparams({
            'ChooseOneOrManyStepsOf__choice': 'a',
        })
        # or
        p.set_hyperparams({
            'ChooseOneStepOf': {
                'a': { 'enabled': True },
                'b': { 'enabled': False }
            }
        })

    .. seealso::
        :class:`Pipeline`
        :class:`Optional`
    """

    def __init__(self, steps, hyperparams=None):
        FeatureUnion.__init__(self, steps, joiner=SelectNonEmptyDataInputs())

        self._make_all_steps_optional()

        if hyperparams is None:
            choices = list(self.keys())[:-1]
            self.set_hyperparams(HyperparameterSamples({
                CHOICE_HYPERPARAM: choices[0]
            }))
            self.set_hyperparams_space(HyperparameterSpace({
                CHOICE_HYPERPARAM: Choice(choices)
            }))

    def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, dict]):
        """
        Set chosen step hyperparams.

        :param hyperparams: hyperparams
        :type hyperparams: HyperparameterSamples
        :return:
        """
        super().set_hyperparams(hyperparams)
        self._update_optional_hyperparams()

        return self

    def update_hyperparams(self, hyperparams: Union[HyperparameterSamples, dict]):
        """
        Set chosen step hyperparams.

        :param hyperparams: hyperparams
        :type hyperparams: HyperparameterSamples
        :return:
        """
        super().update_hyperparams(hyperparams)
        self._update_optional_hyperparams()

        return self

    def _update_optional_hyperparams(self):
        step_names = list(self.keys())
        chosen_step_name = self.hyperparams[CHOICE_HYPERPARAM]
        if chosen_step_name not in step_names:
            raise ValueError('Invalid Chosen Step in {0}'.format(self.name))
        for step_name in step_names[:-1]:
            if step_name == chosen_step_name:
                self[chosen_step_name].set_hyperparams({
                    OPTIONAL_ENABLED_HYPERPARAM: True
                })
            else:
                self[step_name].set_hyperparams({
                    OPTIONAL_ENABLED_HYPERPARAM: False
                })

    def _make_all_steps_optional(self):
        """
        Wrap all steps with :class:`Optional` wrapper.
        """
        step_names = list(self.keys())
        for step_name in step_names[:-1]:
            self[step_name] = Optional(self[step_name].set_name('Optional({})'.format(step_name)),
                                       use_hyperparameter_space=False, nullify_hyperparams=False)

        self._refresh_steps()


class ChooseOneOrManyStepsOf(FeatureUnion):
    """
    A pipeline to allow choosing many steps using an hyperparameter.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            ChooseOneOrManyStepsOf([
                ('a', Identity()),
                ('b', Identity())
            ])
        ])
        p.set_hyperparams({
            'ChooseOneOrManyStepsOf__a__enabled': True,
            'ChooseOneOrManyStepsOf__b__enabled': False
        })
        # or
        p.set_hyperparams({
            'ChooseOneOrManyStepsOf': {
                'a': { 'enabled': True },
                'b': { 'enabled': False }
            }
        })

    .. seealso::
        :class:`Pipeline`
        :class:`Optional`
    """

    def __init__(self, steps):
        FeatureUnion.__init__(self, steps, joiner=NumpyConcatenateOnCustomAxisIfNotEmpty(axis=-1))
        self.set_hyperparams(HyperparameterSamples({}))
        self._make_all_steps_optional()

    def _make_all_steps_optional(self):
        """
        Wrap all steps with :class:`Optional` wrapper.
        """
        step_names = list(self.keys())
        for step_name in step_names[:-1]:
            self[step_name] = Optional(self[step_name])
        self._refresh_steps()


class NumpyConcatenateOnCustomAxisIfNotEmpty(NonFittableMixin, BaseStep):
    """
    Numpy concetenation step where the concatenation is performed along the specified custom axis.
    """

    def __init__(self, axis):
        """
        Create a numpy concatenate on custom axis object.
        :param axis: the axis where the concatenation is performed.
        :return: NumpyConcatenateOnCustomAxis instance.
        """
        self.axis = axis
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def _transform_data_container(self, data_container, context):
        """
        Handle transform.

        :param data_container: the data container to join
        :param context: execution context
        :return: transformed data container
        """
        data_inputs = self.transform([dc.data_inputs for dc in data_container.data_inputs if len(dc.data_inputs) > 0])
        data_container = DataContainer(data_inputs=data_inputs, current_ids=data_container.current_ids,
                                       expected_outputs=data_container.expected_outputs)
        data_container.set_data_inputs(data_inputs)

        return data_container

    def transform(self, data_inputs):
        """
        Apply the concatenation transformation along the specified axis.
        :param data_inputs:
        :return: Numpy array
        """
        return self._concat(data_inputs)

    def _concat(self, data_inputs):
        return np.concatenate(data_inputs, axis=self.axis)


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
            # TODO in kata 2, optional: Add some feature selection right here for the motivated ones:
            #      https://scikit-learn.org/stable/modules/feature_selection.html
            # TODO in kata 2, optional: Add normalization right here (if using other classifiers)
            #      https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
            ChooseOneStepOf([
                SKLearnWrapper(DecisionTreeClassifier(), HyperparameterSpace({
                    'criterion': Choice(['gini', 'entropy']),
                    'splitter': Choice(['best', 'random']),
                    'min_samples_leaf': RandInt(2, 5),
                    'min_samples_split': RandInt(1, 3),
                })),
                SKLearnWrapper(ExtraTreeClassifier(), HyperparameterSpace({
                    'criterion': Choice(['gini', 'entropy']),
                    'splitter': Choice(['best', 'random']),
                    'min_samples_leaf': RandInt(2, 5),
                    'min_samples_split': RandInt(1, 3),
                })),
                Pipeline(
                    [OutputTransformerWrapper(NumpyRavel()), SKLearnWrapper(RidgeClassifier(), HyperparameterSpace({
                        'alpha': Choice([(0.0, 1.0, 10.0), (0.0, 10.0, 100.0)]),
                        'fit_intercept': Boolean(),
                        'normalize': Boolean(),
                    }))]).set_name('RidgeClassifier'),
                Pipeline(
                    [OutputTransformerWrapper(NumpyRavel()), SKLearnWrapper(RidgeClassifierCV(), HyperparameterSpace({
                        'alpha': Choice([(0.0, 1.0, 10.0), (0.0, 10.0, 100.0)]),
                        'fit_intercept': Boolean(),
                        'normalize': Boolean(),
                    }))]).set_name('RidgeClassifierCV'),
                #Pipeline([OutputTransformerWrapper(NumpyRavel()),
                #          SKLearnWrapper(KNeighborsClassifier(), HyperparameterSpace({
                #              'n_neighbors': RandInt(3, 15),
                #              'algorithm': Choice(['auto', 'ball_tree', 'kd_tree', 'brute']),
                #              'leaf_size': RandInt(10, 50)
                #          }))]).set_name('kNeighborsClassifier'),
                Pipeline(
                    [OutputTransformerWrapper(NumpyRavel()), SKLearnWrapper(LogisticRegression(), HyperparameterSpace({
                        'C': LogUniform(0.01, 10.0),
                        'fit_intercept': Boolean(),
                        'dual': Boolean(),
                        'penalty': Choice(['l1', 'l2']),
                        'max_iter': RandInt(20, 200),
                    }))]).set_name('LogisticRegression'),
                Pipeline([OutputTransformerWrapper(NumpyRavel()),
                          SKLearnWrapper(RandomForestClassifier(), HyperparameterSpace({
                              'n_estimators': RandInt(50, 600),
                              'criterion': Choice(['gini', 'entropy']),
                              'min_samples_leaf': RandInt(2, 5),
                              'min_samples_split': RandInt(1, 3),
                              'bootstrap': Boolean(),
                          }))]).set_name('RandomForestClassifier')
            ]),
            TrainOnlyWrapper(NumpyShapePrinter(custom_message="Shape at output after classification")),
            # Shape: [batch_size]
        ]),
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        validation_split_function=validation_splitter(test_size=0.20),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False),
            MetricCallback('accuracy', metric_function=accuracy_score, higher_score_is_better=True)
        ],
        n_trials=6,
        epochs=1,
        hyperparams_repository=InMemoryHyperparamsRepository(cache_folder=cache_folder),
        print_metrics=True,
        refit_trial=True,
    )

    # Do AutoML and get best model refitted:
    X_train, y_train, X_test, y_test = load_all_data()

    auto_ml = auto_ml.fit(X_train, y_train)
    pipeline = auto_ml.get_best_model()

    # Predict on test data and score:
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Test accuracy score:", accuracy)
    assert accuracy > 0.90


if __name__ == '__main__':
    main()
