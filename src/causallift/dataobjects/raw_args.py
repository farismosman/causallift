class RawArgs:
    def __init__(self, 
        cols_features, col_treatment, col_outcome, col_propensity, 
        col_cate, col_recommendation, min_propensity, max_propensity,
        verbose, uplift_model_params, enable_ipw, propensity_model_params, 
        index_name, partition_name, runner, conditionally_skip):

        self.cols_features = cols_features
        self.col_treatment = col_treatment
        self.col_outcome = col_outcome
        self.col_propensity = col_propensity
        self.col_cate = col_cate
        self.col_recommendation = col_recommendation
        self.min_propensity = min_propensity
        self.max_propensity = max_propensity
        self.verbose = verbose
        self.uplift_model_params = uplift_model_params
        self.enable_ipw = enable_ipw
        self.propensity_model_params = propensity_model_params
        self.index_name = index_name
        self.partition_name = partition_name
        self.runner = runner
        self.conditionally_skip = conditionally_skip
