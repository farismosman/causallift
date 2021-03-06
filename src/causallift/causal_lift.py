from .nodes.estimate_propensity import (
    bundle_train_and_test_data, impute_cols_features, schedule_propensity_scoring,
    treatment_fractions_, treatment_fractions_, fit_propensity, estimate_propensity,
    compute_cate, add_cate_to_df, recommend_by_cate, estimate_effect)

from .nodes.model_for_each import (
    model_for_treated_simulate_recommendation, model_for_untreated_simulate_recommendation,
    model_for_treated_fit, model_for_untreated_fit, bundle_treated_and_untreated_models,
    model_for_treated_predict_proba, model_for_untreated_predict_proba)

from .base_causal_lift import BaseCausalLift


class CausalLift(BaseCausalLift):
   
    def __init__(self, train_df, test_df, **kwargs):
        super(CausalLift, self).__init__(**kwargs)
        self.runner = None
        
        self.df = bundle_train_and_test_data(self.args_raw, train_df, test_df)
        self.args = impute_cols_features(self.args_raw, self.df)
        self.args = schedule_propensity_scoring(self.args, self.df)
        self.treatment_fractions = treatment_fractions_(self.args, self.df)
        if self.args.need_propensity_scoring:
            self.propensity_model = fit_propensity(self.args, self.df)
            self.df = estimate_propensity(self.args, self.df, self.propensity_model)

        self.treatment_fraction_train = self.treatment_fractions.train
        self.treatment_fraction_test = self.treatment_fractions.test

        self._separate_train_test()

    def estimate_cate_by_2_models(self):
        treated__model_dict = model_for_treated_fit(self.args, self.df)
        untreated__model_dict = model_for_untreated_fit(self.args, self.df)
        self.uplift_models_dict = bundle_treated_and_untreated_models(treated__model_dict, untreated__model_dict)

        self.treated__proba = model_for_treated_predict_proba(self.args, self.df, self.uplift_models_dict)
        self.untreated__proba = model_for_untreated_predict_proba(self.args, self.df, self.uplift_models_dict)
        self.cate_estimated = compute_cate(self.treated__proba, self.untreated__proba)
        self.df = add_cate_to_df(self.args, self.df, self.cate_estimated)

        return self._separate_train_test()

    def estimate_recommendation_impact(
        self, cate_estimated=None, treatment_fraction_train=None, 
        treatment_fraction_test=None, verbose=None):

        super(CausalLift, self).estimate_recommendation_impact(
            cate_estimated,
            treatment_fraction_train,
            treatment_fraction_test,
            verbose
        )

        self.treatment_fractions.train = (treatment_fraction_train or self.treatment_fractions.train)
        self.treatment_fractions.test = (treatment_fraction_test or self.treatment_fractions.test)

        self.df = recommend_by_cate(self.args, self.df, self.treatment_fractions)
        self.treated__sim_eval_df = model_for_treated_simulate_recommendation(self.args, self.df, self.uplift_models_dict)
        self.untreated__sim_eval_df = model_for_untreated_simulate_recommendation(self.args, self.df, self.uplift_models_dict)
        self.estimated_effect_df = estimate_effect(self.treated__sim_eval_df, self.untreated__sim_eval_df)