from IPython.display import display
import numpy as np
import pandas as pd

from .utils import (initialize_model, score_df, concat_train_test)
from causallift.base_causal_lift import log
from .model_for_each import ModelForTreatedOrUntreated


class IModelForTreatedOrUntreated(ModelForTreatedOrUntreated):
    def __init__(self, treatment_val=1.0):
        super(IModelForTreatedOrUntreated, self).__init__(treatment_val=treatment_val)

    def fit(self, args, df_):
        output = super(IModelForTreatedOrUntreated, self).fit(args, df_)

        self._display_model_info()
        self._display_best_parameters(output['model'])
        self._display_feature_importance(output['model'])
        self._display_model_outcome(output['eval_df'])

        return output

    def simulate_recommendation(self, args, df_, models_dict):
        out_df = super(IModelForTreatedOrUntreated, self).simulate_recommendation(args, df_, models_dict)
        
        self._display_recommended_output(args, out_df)

        return out_df

    def _display_model_info(self):
        log.info("## Model for Treatment = {}".format(self.treatment_val))

    def _display_propensity_warnings(self, args, propensity):
        if propensity.min() < args.min_propensity:
            log.warn(
                "Propensity scores below {} were clipped.".format(
                    args.min_propensity
                )
            )
        if propensity.max() > args.max_propensity:
            log.warn(
                "Propensity scores above {} were clipped.".format(
                    args.max_propensity
                )
            )

    def _display_best_parameters(self, model):
        log.debug(
                "### Best parameters of the model trained using samples "
                "with observational Treatment: {} \n {}".format(
                    self.treatment_val, model.best_params_
                )
            )

    def _display_feature_importance(self, model):
        if hasattr(model.best_estimator_, "feature_importances_"):
            fi_df = pd.DataFrame(
                model.best_estimator_.feature_importances_.reshape(1, -1),
                index=["feature importance"],)
            log.info(
                "### Feature importances of the model trained using samples "
                "with observational Treatment: {}".format(self.treatment_val)
            )
            display(fi_df)
        else:
            log.info("## Feature importances not available.")

    def _display_model_outcome(self, score_original_treatment_df):
        log.debug(
            "### Outcome estimated by the model trained using samples "
            "with observational Treatment: {}".format(self.treatment_val)
        )
        display(score_original_treatment_df)

    def _display_recommended_output(self, args, df):
        log.info("### Simulated outcome of samples recommended to be treatment: {} by the uplift model:".format(self.treatment_val))
        if args.verbose >= 3:
            display(df)

class IModelForTreated(IModelForTreatedOrUntreated):
    def __init__(self, *posargs, **kwargs):
        kwargs.update(treatment_val=1.0)
        super().__init__(*posargs, **kwargs)


class IModelForUntreated(IModelForTreatedOrUntreated):
    def __init__(self, *posargs, **kwargs):
        kwargs.update(treatment_val=0.0)
        super().__init__(*posargs, **kwargs)


def model_for_treated_fit(*posargs, **kwargs):
    return IModelForTreated().fit(*posargs, **kwargs)


def model_for_treated_predict_proba(*posargs, **kwargs):
    return IModelForTreated().predict_proba(*posargs, **kwargs)


def model_for_treated_simulate_recommendation(*posargs, **kwargs):
    return IModelForTreated().simulate_recommendation(*posargs, **kwargs)


def model_for_untreated_fit(*posargs, **kwargs):
    return IModelForUntreated().fit(*posargs, **kwargs)


def model_for_untreated_predict_proba(*posargs, **kwargs):
    return IModelForUntreated().predict_proba(*posargs, **kwargs)

def model_for_untreated_simulate_recommendation(*posargs, **kwargs):
    return IModelForUntreated().simulate_recommendation(*posargs, **kwargs)

def bundle_treated_and_untreated_models(treated_model, untreated_model):
    models_dict = dict(treated=treated_model, untreated=untreated_model)
    return models_dict