import numpy as np
import importlib

from sklearn import model_selection


class PrunedCV:


    def __init__(self, folds, X_train, y_train):

        self.folds = folds
        self.X_train = X_train
        self.y_train = y_train


    def set_params(self, param_grid, scores):

        self.param_grid = param_grid
        self.scores = scores


    def set_threshold(self, THRESH_SKIP, THRESH_PERCENTAGE, SCORE, N_FOLDS):

        self.THRESH_SKIP = THRESH_SKIP
        self.THRESH_PERCENTAGE = THRESH_PERCENTAGE
        self.SCORE = SCORE
        self.N_FOLDS = N_FOLDS
    

    def evaluate_model(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
                       model: type, scores: list) -> dict:
        '''
        This helper function, given a model and the data, allows us to compute all the scores of interest.

        **Args**\n
            "X_train": train set.\n
            "y_train": train labels.\n
            "X_test": test set.\n
            "y_test": test labels.\n
            "model": model we want to evaluate. It should be a class like naive_bayes.BernoulliNB or svm.svc.\n
            "scores": a list of scores we want to compute. For example, accuracy_score of f1_score.\n
        
        **Returns**\n
            Dictionary containing the computed scores.
        '''

        model.fit(X_train, y_train)                  
        y_hat = model.predict(X_test)    

        # We exploit list comprehension to compute each score.            
        results = [score(y_test, y_hat) for score in scores]
        dic_results =  {k: v for k, v in zip([score.__name__ for score in scores], results)}
        
        return dic_results


    def do_cross_validation(self, prune = True):

        self.models_performance = {}
        # We initialize the maximum performance. It will be used as a benchmark.
        skip_max = 0


        for model_str in self.param_grid.keys():                       # {'svc': None} <-- MODEL LEVEL

            count_config = 0                                    # ith-configuration.
            model_name   = model_str.split('.')[-1] # Global name of the model we are evaluating.

            # Used to get the method and not just the string.
            module = importlib.import_module(".".join(model_str.split('.')[:-1]))     #########################
            model  = getattr(module, model_name)                                      # || MODEL ITERATION || #
                                                                                    #########################
            self.models_performance[model_name] = {}
            print(f"Model: {model_str}\n")
            
            # We iterate over all the possible configuration.
            for config in model_selection.ParameterGrid(self.param_grid[model_str]):                                                                         
                                                        
                print("\n\tNEW CONFIGURATION")
                model_config_name = model_name + f"_{count_config}"                                                   #################################
                count_instance    = 0   # ith fold.                                                                   # || CONFIGURATION ITERATION || #
                self.models_performance[model_name][model_config_name] = {score: [] for score in [score.__name__ for score in self.scores]}
                self.models_performance[model_name][model_config_name]['weight'] = []                                                                                                  ################################# 
                print(f"\nConfiguration: {config}\n")                
                
                skip = False   
                count_skip = 0                                                                                     
                count_fold = 1

                # We can start the evaluation of the model using the folds.
                for train_indices, valid_indices in self.folds.split(self.X_train, self.y_train):
                    
                    if count_skip >= self.THRESH_SKIP and prune:

                        break
                                                                    
                    X_train_fold = self.X_train[train_indices]      
                    y_train_fold = self.y_train[train_indices]      
                    X_valid_fold = self.X_train[valid_indices]      
                    y_valid_fold = self.y_train[valid_indices]
                                                                                    ############################
                    clf = model(**config)                                           # || INSTANCE ITERATION || #
                                                                                    ############################
                        
                    model_instance_name = model_config_name + f"_{count_instance}"
                    results = self.evaluate_model(X_train_fold, y_train_fold, X_valid_fold, y_valid_fold,
                                                  clf, self.scores)
                    
                    for score, result in results.items():
                        
                        self.models_performance[model_name][model_config_name][score].append(result)


                    # Here we store the weight of the score since we may have different amounts of samples.
                    self.models_performance[model_name][model_config_name]['weight'].append(len(train_indices))
                    dummy_1 = self.models_performance[model_name][model_config_name][self.SCORE]
                    dummy_2 = self.models_performance[model_name][model_config_name]['weight']
                    
            
                    if np.average(dummy_1, weights = dummy_2) < self.THRESH_PERCENTAGE * skip_max and prune:

                        count_skip += 1
                    
                    elif np.average(dummy_1, weights = dummy_2) >= skip_max and count_fold >= self.N_FOLDS and prune:

                        skip_max = results[self.SCORE]
                    
                    print(f"Fold {count_fold} / {self.folds.get_n_splits()} - Skip: {count_skip} / {self.THRESH_SKIP}")
                    print(f"Results: {results}")

                    count_fold += 1
                    print(skip_max)

                # We store also the parameters in order to retrieve the best configuration eventually.
                self.models_performance[model_name][model_config_name]['parameters'] = config
                count_config += 1
                self.models_performance[model_name][model_config_name]['skip'] = skip

            print('\n')