import optuna



def objective(trial):
    pass

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=config['hyperopt_trail'])