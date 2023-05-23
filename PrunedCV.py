import numpy as np
import importlib

from sklearn import model_selection


class PrunedCV:


    def __init__(self, folds, X_train, y_train):

        self.__folds__   = folds
        self.__X_train__ = X_train
        self.__y_train__ = y_train


    def set_params(self, param_grid, scores):

        self.__param_grid__ = param_grid
        self.__scores__     = scores


    def set_threshold(self, THRESH_SKIP, THRESH_PERCENTAGE, SCORE, N_FOLDS):

        self.__THRESH_SKIP__       = THRESH_SKIP
        self.__THRESH_PERCENTAGE__ = THRESH_PERCENTAGE
        self.__SCORE__             = SCORE
        self.__N_FOLDS__           = N_FOLDS
    

    def __evaluate_model__(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
                           model: type, scores: list) -> dict:
        '''
        Helper function that, given a model and the data, allows us to compute all the scores of interest.

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


    def do_cross_validation(self, prune = True, verbose = 2):

        models_performance = {}
        # We initialize the maximum performance. It will be used as a benchmark.
        skip_max = 0


        for model_str in self.__param_grid__.keys():                       # {'svc': None} <-- MODEL LEVEL

            count_config = 0                                    # ith-configuration.
            model_name   = model_str.split('.')[-1] # Global name of the model we are evaluating.

            # Used to get the method and not just the string.
            module = importlib.import_module(".".join(model_str.split('.')[:-1]))     #########################
            model  = getattr(module, model_name)                                      # || MODEL ITERATION || #
                                                                                    #########################
            models_performance[model_name] = {}
            print(f"Model: {model_str}\n")
            
            # We iterate over all the possible configuration.
            for config in model_selection.ParameterGrid(self.__param_grid__[model_str]):                                                                         
                                                        
                print("\n\tNEW CONFIGURATION")
                model_config_name = model_name + f"_{count_config}"                                                   #################################
                count_instance    = 0   # ith fold.                                                                   # || CONFIGURATION ITERATION || #
                models_performance[model_name][model_config_name] = {score: [] for score in [score.__name__ for score in self.__scores__]}
                models_performance[model_name][model_config_name]['weight'] = []                                                                                                  ################################# 
                
                if verbose >= 1:

                    print(f"\nConfiguration: {config}\n")                
                
                skip = False   
                count_skip = 0                                                                                     
                count_fold = 1

                # We can start the evaluation of the model using the folds.
                for train_indices, valid_indices in self.__folds__.split(self.__X_train__, self.__y_train__):
                    
                    if count_skip >= self.__THRESH_SKIP__ and prune:

                        break
                                                                    
                    X_train_fold = self.__X_train__[train_indices]      
                    y_train_fold = self.__y_train__[train_indices]      
                    X_valid_fold = self.__X_train__[valid_indices]      
                    y_valid_fold = self.__y_train__[valid_indices]
                                                                                    ############################
                    clf = model(**config)                                           # || INSTANCE ITERATION || #
                                                                                    ############################
                        
                    model_instance_name = model_config_name + f"_{count_instance}"
                    results = self.__evaluate_model__(X_train_fold, y_train_fold, X_valid_fold, y_valid_fold,
                                                  clf, self.__scores__)
                    
                    for score, result in results.items():
                        
                        models_performance[model_name][model_config_name][score].append(result)


                    # Here we store the weight of the score since we may have different amounts of samples.
                    models_performance[model_name][model_config_name]['weight'].append(len(train_indices))
                    dummy_1 = models_performance[model_name][model_config_name][self.__SCORE__]
                    dummy_2 = models_performance[model_name][model_config_name]['weight']
                    
            
                    if np.average(dummy_1, weights = dummy_2) < self.__THRESH_PERCENTAGE__ * skip_max and prune:

                        count_skip += 1
                    
                    elif np.average(dummy_1, weights = dummy_2) >= skip_max and count_fold >= self.__N_FOLDS__ and prune:

                        skip_max = results[self.__SCORE__]
                    
                    if verbose == 2:
                        
                        print(f"Fold {count_fold} / {self.__folds__.get_n_splits()} - Skip: {count_skip} / {self.__THRESH_SKIP__}")
                        print(f"Results: {results}")
                        print(f"Highest average {self.__SCORE__}: {np.round(skip_max, 4)}")

                    count_fold += 1

                # We store also the parameters in order to retrieve the best configuration eventually.
                models_performance[model_name][model_config_name]['parameters'] = config
                count_config += 1
                models_performance[model_name][model_config_name]['skip'] = skip

            print('\n')
    
        return models_performance
