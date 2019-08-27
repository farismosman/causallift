from kedro.io import MemoryDataSet
from causallift.context.flexible_context import FlexibleKedroContext

from .nodes.estimate_propensity import (
    bundle_train_and_test_data, impute_cols_features, schedule_propensity_scoring,
    treatment_fractions_, treatment_fractions_, fit_propensity, estimate_propensity,
    compute_cate, add_cate_to_df, recommend_by_cate, estimate_effect)

from .nodes.model_for_each import (
    model_for_treated_simulate_recommendation, model_for_untreated_simulate_recommendation,
    model_for_treated_fit, model_for_untreated_fit, bundle_treated_and_untreated_models,
    model_for_treated_predict_proba, model_for_untreated_predict_proba)

from .base_causal_lift import BaseCausalLift



class KedroCausalLift(BaseCausalLift):

    def __init__(self, train_df, test_df, **kwargs):
        super(KedroCausalLift, self).__init__(**kwargs)

        assert self.runner in {"SequentialRunner", "ParallelRunner"}

        self.kedro_context = FlexibleKedroContext(runner=self.runner, only_missing=self.args_raw.conditionally_skip)

        self.kedro_context.catalog.add_feed_dict(
            {
                "train_df": MemoryDataSet(train_df),
                "test_df": MemoryDataSet(test_df),
                "args_raw": MemoryDataSet(self.args_raw),
            },
            replace=True,
        )
        self.kedro_context.catalog.add_feed_dict(self.dataset_catalog, replace=True)

        self.kedro_context.run(tags=["011_bundle_train_and_test_data"])
        self.df = self.kedro_context.catalog.load("df_00")

        self.kedro_context.run(
            tags=[
                "121_prepare_args",
                "131_treatment_fractions_",
                "141_initialize_model",
            ]
        )
        self.args = self.kedro_context.catalog.load("args")
        self.treatment_fractions = self.kedro_context.catalog.load("treatment_fractions")

        if self.args.need_propensity_scoring:
            self.kedro_context.run(tags=["211_fit_propensity"])
            self.propensity_model = self.kedro_context.catalog.load("propensity_model")
            self.kedro_context.run(tags=["221_estimate_propensity"])
            self.df = self.kedro_context.catalog.load("df_01")
        else:
            self.kedro_context.catalog.add_feed_dict(
                {"df_01": MemoryDataSet(self.df)},
                replace=True
            )

        self.treatment_fraction_train = self.treatment_fractions.train
        self.treatment_fraction_test = self.treatment_fractions.test

        self._separate_train_test()


    def estimate_cate_by_2_models(self):

        self.kedro_context.run(tags=["311_fit", "312_bundle_2_models"])
        self.uplift_models_dict = self.kedro_context.catalog.load("uplift_models_dict")

        self.kedro_context.run(tags=["321_predict_proba"])
        self.treated__proba = self.kedro_context.catalog.load("treated__proba")
        self.untreated__proba = self.kedro_context.catalog.load("untreated__proba")
        self.kedro_context.run(tags=["411_compute_cate"])
        self.cate_estimated = self.kedro_context.catalog.load("cate_estimated")
        self.kedro_context.run(tags=["421_add_cate_to_df"])
        self.df = self.kedro_context.catalog.load("df_02")

        return self._separate_train_test()

    def estimate_recommendation_impact(
        self, cate_estimated=None, treatment_fraction_train=None, 
        treatment_fraction_test=None, verbose=None
        ):

        super(KedroCausalLift, self).estimate_recommendation_impact(
            cate_estimated,
            treatment_fraction_train,
            treatment_fraction_test,
            verbose
        )

        self.treatment_fractions.train = (treatment_fraction_train or self.treatment_fractions.train)
        self.treatment_fractions.test = (treatment_fraction_test or self.treatment_fractions.test)

        # self.kedro_context.catalog.save('args', self.args)
        self.kedro_context.run(tags=["511_recommend_by_cate"])
        self.df = self.kedro_context.catalog.load("df_03")

        self.kedro_context.run(tags=["521_simulate_recommendation"])
        self.treated__sim_eval_df = self.kedro_context.catalog.load("treated__sim_eval_df")
        self.untreated__sim_eval_df = self.kedro_context.catalog.load("untreated__sim_eval_df")

        self.kedro_context.run(tags=["531_estimate_effect"])
        self.estimated_effect_df = self.kedro_context.catalog.load("estimated_effect_df")

        return self.estimated_effect_df
