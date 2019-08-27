from kedro.io import CSVLocalDataSet, PickleLocalDataSet, MemoryDataSet


class DatasetCatalog:
    def __init__(self, 
        propensity_model_filename="../data/06_models/propensity_model.pickle",
        uplift_models_filename="../data/06_models/uplift_models_dict.pickle",
        df_filename="../data/07_model_output/df.csv",
        treated_sim_eval_filename="../data/08_reporting/treated__sim_eval_df.csv",
        untreated_sim_eval_filename="../data/08_reporting/untreated__sim_eval_df.csv",
        estimated_effect_filename="../data/08_reporting/estimated_effect_df.csv",
        args_raw=MemoryDataSet({}).load()):

        self.propensity_model = PickleLocalDataSet(filepath=propensity_model_filename, version=None)
        self.uplift_models_dict = PickleLocalDataSet(filepath=uplift_models_filename, version=None)
        self.df_03 = CSVLocalDataSet(
            filepath=df_filename,
            load_args=dict(
                index_col=["partition", "index"], float_precision="high"
            ),
            save_args=dict(index=True, float_format="%.16e"),
            version=None,
        )
        self.treated__sim_eval_df = CSVLocalDataSet(filepath=treated_sim_eval_filename, version=None)
        self.untreated__sim_eval_df = CSVLocalDataSet(filepath=untreated_sim_eval_filename, version=None)
        self.estimated_effect_df = CSVLocalDataSet(filepath=estimated_effect_filename, version=None)
        self.args_raw = args_raw