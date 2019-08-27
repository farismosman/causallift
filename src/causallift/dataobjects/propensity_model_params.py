class PropensityModelParams:
    def __init__(self, 
        search_cv="sklearn.model_selection.GridSearchCV",
        estimator="sklearn.linear_model.LogisticRegression",
        scoring=None,
        cv=3,
        return_train_score=False,
        n_jobs=-1,
        param_grid=dict(
            random_state=[0],
            C=[0.1, 1, 10],
            class_weight=[None],
            dual=[False],
            fit_intercept=[True],
            intercept_scaling=[1],
            max_iter=[100],
            multi_class=["ovr"],
            n_jobs=[1],
            penalty=["l1", "l2"],
            solver=["liblinear"],
            tol=[0.0001],
            warm_start=[False]
        )):

        self.search_cv = search_cv
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.return_train_score = return_train_score
        self.n_jobs = n_jobs
        self.param_grid = param_grid
