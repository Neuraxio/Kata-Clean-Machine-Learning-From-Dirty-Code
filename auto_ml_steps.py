from typing import Union

import numpy as np
from neuraxle.base import ForceHandleOnlyMixin, MetaStepMixin, BaseStep, ExecutionContext, NonFittableMixin
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.distributions import Choice, Boolean
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.steps.flow import SelectNonEmptyDataInputs, CHOICE_HYPERPARAM, OPTIONAL_ENABLED_HYPERPARAM
from neuraxle.union import FeatureUnion


class Optional(ForceHandleOnlyMixin, MetaStepMixin, BaseStep):
    """
    A wrapper to nullify a step : nullify its hyperparams, and also nullify all of his behavior.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            Optional(Identity(), enabled=True)
        ])

    """

    def __init__(self, wrapped: BaseStep, enabled: bool = True, nullified_return_value=None, cache_folder_when_no_handle=None, use_hyperparameter_space=True, nullify_hyperparams=True):
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

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
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

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
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
            self[step_name] = Optional(self[step_name].set_name('Optional({})'.format(step_name)), use_hyperparameter_space=False, nullify_hyperparams=False)

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
