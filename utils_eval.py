import numpy as np
import pandas as pd

def get_feature_importance(gbm_booster):
    feat_ind = gbm_booster.feature_importance().argsort()[::-1]

    feat_names_sorted = np.array(gbm_booster.feature_name())[feat_ind]
    feat_imp_sorted = np.array(gbm_booster.feature_importance())[feat_ind]

    for ind in range(feat_names_sorted.shape[0]):
        print(feat_names_sorted[ind] + ": " + str(feat_imp_sorted[ind]))


def get_act_pred(preds_df):
    print("% correctly predicted: " + 
        str(round(sum(preds_df.pred == preds_df.act)/preds_df.shape[0], 4))
    )

    print("% Home Wins correctly predicted: " + 
        str(round(sum(preds_df[preds_df.act == "W"].pred == preds_df[preds_df.act == "W"].act)/preds_df[preds_df.act == "W"].shape[0], 4))
    )

    print("% Away Wins correctly predicted: " + 
        str(round(sum(preds_df[preds_df.act == "L"].pred == preds_df[preds_df.act == "L"].act)/preds_df[preds_df.act == "L"].shape[0], 4))
    )

    print("% Draws correctly predicted: " + 
        str(round(sum(preds_df[preds_df.act == "D"].pred == preds_df[preds_df.act == "D"].act)/preds_df[preds_df.act == "D"].shape[0], 4))
    )
