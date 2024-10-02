
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

    ip_path            =f'/mnt/c/Users/anush/Documents/Competitions/BrisT1D Blood Glucose Prediction'
    op_path            =ip_path

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
