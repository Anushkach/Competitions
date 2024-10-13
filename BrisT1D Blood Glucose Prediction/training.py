
# Configuration class
class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    """

    # Data preparation
    version_nb         =1
    model_id           ="V1_2"
    model_label        ="ML"

    test_req           =False
    test_sample_frac   =1000

    gpu_switch         ="OFF"
    state              =42
    target             =f"bg+1:00"
    grouper            =f"p_num"

    ip_path            =f'/mnt/c/Users/anush/Documents/Competitions/BrisT1D Blood Glucose Prediction/inputs'
    op_path            =f'/mnt/c/Users/anush/Documents/Competitions/BrisT1D Blood Glucose Prediction/outputs'

    # Model Training
    pstprcs_oof        =False
    pstprcs_train      =False
    pstprcs_test       =False
    ML                 =True
    test_preds_req     =True

    pseudo_lbl_req      =False
    pseudolbl_up        =0.975
    pseudolbl_low       =0.0

    n_splits        =3 if test_req else 5
    n_repeats       =1
    nbrnd_erly_stp  =100
    mdlcv_mthd      ='GKF'

    #Ensemble
    ensemble_req=True
    metric_obj='minimize'

    # Global variables for plotting
    grid_specs={
        'visible':True,
        'which':"both",
        "linestyle":"--",
        "color":"lightgrey",
        "linewidth":0.75
        }
    
    title_specs={
        'fontsize':9,
        'fontweight':"bold",
        'color':"#992600"
                 }
cv_selector=\
{
    "RKF":RKF(n_splits=CFG.n_splits,n_repeats=CFG.n_repeats,random_state=CFG.state),
    "RSKF":RSKF(n_splits=CFG.n_splits,n_repeats=CFG.n_repeats,random_state=CFG.state),
    "SKF":SKF(n_splits=CFG.n_splits,shuffle=True,random_state=CFG.state),
    "KF":KFold(n_splits=CFG.n_splits,shuffle=True,random_state=CFG.state),
    "GKF":GKF(n_splits=CFG.n_splits)
}

PrintColor(f"\n---> Configuration done !\n")
collect()

def XformCols(df:pd.DataFrame):
    """
    This function do the following.
    * Create time components on time column
    * Removes special characters in column labels

    """

    df["hour_nb"]   =pd.to_datetime(df['time']).dt.hour
    df["minute_nb"] =pd.to_datetime(df['time']).dt.minute
    df.columns      =df.columns.str.replace("\W",'',regex=True)

    return df.drop("time", axis=1,errors='ignore')


class ModelTrainer:
    """ 
    This Class trains the provided model on train-test data and returns the predictions and fitted models
    """

    def __init__(self,
                 es_req:bool=False,
                 es:int=100,
                 target:str=CFG.target,
                 metric_lbl:str='rmse',
                 drop_cols:list=["Source","id","Id","Label",CFG.target,"fold_nb",CFG.grouper],
                 ):
        """ 
        key parameters-
        es_iter: early stopping rounds for boosted trees
        """
        drop_cols=list(set(drop_cols+[target]))

        self.es_req=es_req
        self.es_iter=es
        self.target=target
        self.drop_cols=drop_cols
        self.metric_lbl=metric_lbl

    def ScoreMetric(self,ytrue,ypred)->float:
        """ 
        This is the metric function for the competition scoring
        """
        return rmse(ytrue,ypred)
    
    def PlotFtreImp(self,
                    ftreimp:pd.Series,
                    method:str,
                    ntop:int=50,
                    title_specs:dict=CFG.title_specs,
                    **params,
                    ):
        """ 
        This functon plots the feature importance for the model provided
        """
        print()
        fig,ax=plt.subplots(1,1,figsize=(25,8))
        ftreimp.sort_values(ascending=False).head(ntop).plot.bar(ax=ax,color="blue")
        ax.set_title(f"Fearture Importances -{method}",**title_specs)

        plt.tight_layout()
        plt.show()

        print()

    def PostProcessPreds(self,ypred):
        """ 
        This method post-processes the predictions optionally
        """
        return np.clip(ypred,a_min=0,a_max=np.inf)
    

    def MakeOfflineModel(self,
                         X,y,ygrp,Xttest,mdl,method,
                         test_preds_req:bool=True,
                         ftreimp_plot_req:bool=True,
                         ntop:int=50,
                         **params):
        """
        This method trains the provided model on the dataset and cross validates appropriately

        Inputs-
        X,y,ygrp        -train data components
        Xtest           -test data
        model           -model object for training
        method          -model method label
        test_preds_req  -boolean flag to extract test set predictions
        ftreimp_req     -boolean falg to plot tree feature importances
        ntop            -top n features for feature importances plot

        Returns
        oof_preds,test_preds    -prediction arrays
        fitted_models           -fitted model list for test set
        ftreimp                 -feature importance across selected features
        mdl_best_iter           -model average best iteration across folds

        """
        oof_preds=np.zeros(len(X))
        test_preds=[]
        mdl_best_iter=[]
        ftreimp=0

        scores,tr_scores,fitted_models=[],[],[]
        cv=PDS(ygrp)
        n_splits=ygrp.nunique()

        for fold_nb,(train_idx,dev_idx) in tqdm(enumerate(cv.split(X,y))):
            Xtr=X.iloc[train_idx].drop("Source",axis=1,errors="ignore")
            Xdev=X.iloc[dev_idx].query("Source=='Competition'").drop("Source",axis=1,errors="ignore")
            ytr=y.iloc[Xtr.index]
            ydev=y.iloc[Xdev.index]
            model=clone(mdl)

            if "CB" in method and self.es_req==True:
                model.fit(Xtr,ytr,
                          eval_set=[(Xdev,ydev)],
                          verbose=0,
                          early_stopping_rounds=self.es_iter
                          )
                best_iter=model.get_best_iteration()

            elif "LGB" in method and self.es_req==True:
                model.fit(Xtr,ytr,
                          eval_set=[(Xdev,ydev)],
                          callbacks=[log_evaluation(0),
                                      early_stopping(stopping_rounds=self.es_iter,
                                                     verbose=False),
                                                ],
                        )
                best_iter=model.best_iteration_

            elif "XGB" in method and self.es_req==True:
                model.fit(Xtr,ytr,
                          eval_set=[(Xdev,ydev)],
                          verbose=0,
                          )
                best_iter=model.best_iteration

            else:
                model.fit(Xtr,ytr)
                best_iter=-1

            fitted_models.append(model)
            
            try:
                ftreimp+=model.feature_importances_
            except:
                pass

            dev_preds=self.PostProcessPreds(model.predict(Xdev))
            oof_preds[Xdev.index]=dev_preds

            train_preds=self.PostProcessPreds(model.predict(Xtr))
            tr_score=self.ScoreMetric(ytr.values.flatten(),train_preds)
            score=self.ScoreMetric(ydev,dev_preds)

            scores.append(score)
            tr_scores.append(tr_score)

            nspace=15-len(method)-2 if fold_nb<=9 else 15-len(method)-1

            if self.es_req:
                PrintColor(f"{method} Fold{fold_nb} {' '*nspace} OOF = {score:.6f} | Train = {tr_score:.6f} | Iter = {best_iter:,.0f}")
            else:
                PrintColor(f"{method} Fold{fold_nb} {' '*nspace} OOF = {score:.6f} | Train = {tr_score:.6f}")

            mdl_best_iter.append(best_iter)

            if test_preds_req:
                test_preds.append(self.PostProcessPreds(model.predict(Xtest)))
            else:
                pass

        tets_preds=np.mean(np.stack(test_preds,axis=1),axis=1) 
        ftreimp=pd.Series(ftreimp,index=Xdev.columns)
        mdl_best_iter=np.uint16(np.amax(mdl_best_iter))

        if ftreimp_plot_req==True and best_iter>0:
            print()
            self.PlotFtreImp(ftreimp,method=method,ntop=ntop)
        else:
            pass

        PrintColor(f"\n---> {np.mean(scores):.6f} +- {np.std(scores):.6f} | OOF", color=Fore.RED)
        PrintColor(f"---> {np.mean(tr_scores):.6f} +- {np.std(tr_scores):.6f} | OOF",color=Fore.RED)
        
        if self.es_req==False:
            pass
        else:
            PrintColor(f"---> Max best iteration = {mdl_best_iter:,.0f}",color=Fore.RED)

        return (fitted_models,oof_preds,test_preds,ftreimp,mdl_best_iter)
    
    def MakeOnlineModel(self,X,y,Xtest,model,method,
                        test_preds_req:bool=False):
        """ 
        This method refits the model on complete train data and returns the model fitted object and predictions
        """
        try:
            model.early_stopping_rounds=None
        except:
            pass

        try:
            model.fit(X,y,verbose=0)
        except:
            model.fit(X,y)
        
        oof_preds=model.predict(X)

        if test_preds_req:
            test_preds=model.predict(Xtest[X.columns])
        else:
            test_preds=0
        return (model,oof_preds,test_preds)
    
    def MakeOfflinePreds(self,X,fitted_models):
        """
        This method creates test-set predictions for the offline model provided
        """
        test_preds=0
        n_splits=len(fitted_models)
        PrintColor(f"---> Number of splits = {n_splits}")

        for model in fitted_models:
            test_preds+=model.predict(X)/n_splits

        return test_preds
    

