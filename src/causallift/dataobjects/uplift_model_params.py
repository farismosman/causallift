class UpliftModelParams:
    def __init__(self, 
        search_cv="sklearn.model_selection.GridSearchCV",
        estimator="xgboost.XGBClassifier",
        scoring=None,
        cv=3,
        return_train_score=False,
        n_jobs=-1,
        param_grid=dict(
            random_state=[0],
            max_depth=[3],
            learning_rate=[0.1],
            n_estimators=[100],
            verbose=[0],
            objective=["binary:logistic"],
            booster=["gbtree"],
            n_jobs=[-1],
            nthread=[None],
            gamma=[0],
            min_child_weight=[1],
            max_delta_step=[0],
            subsample=[1],
            colsample_bytree=[1],
            colsample_bylevel=[1],
            reg_alpha=[0],
            reg_lambda=[1],
            scale_pos_weight=[1],
            base_score=[0.5],
            missing=[None]
        )):

        self.search_cv = search_cv
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.return_train_score = return_train_score
        self.n_jobs = n_jobs
        self.param_grid = param_grid
