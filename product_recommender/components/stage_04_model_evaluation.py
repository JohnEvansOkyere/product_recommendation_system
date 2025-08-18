# product_recommender/components/evaluate_model.py
import os
import sys
import argparse
import pickle
import numpy as np
import scipy.sparse as sp
from product_recommender.logger.log import logging
from product_recommender.config.configuration import AppConfiguration
from product_recommender.exception.exception_handler import AppException
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k


class ModelEvaluator:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.config = app_config.get_model_evaluation_config()
            self.model_dir = self.config.model_dir
        except Exception as e:
            raise AppException(e, sys) from e

    def sampled_eval(self, model, interactions, user_features, item_features, k=10, sample_users=5000, num_threads=4):
        try:
            # convert to csr for slicing
            if not hasattr(interactions, "tocsr"):
                interactions = interactions.tocsr()
            n_users, _ = interactions.shape
            if sample_users and sample_users < n_users:
                idx = np.random.choice(n_users, size=sample_users, replace=False)
                inter_sub = interactions[idx, :]
                user_feat_sub = user_features[idx, :] if user_features is not None else None
                auc = auc_score(model, inter_sub, user_features=user_feat_sub, item_features=item_features, num_threads=num_threads).mean()
                prec = precision_at_k(model, inter_sub, k=k, user_features=user_feat_sub, item_features=item_features, num_threads=num_threads).mean()
                rec = recall_at_k(model, inter_sub, k=k, user_features=user_feat_sub, item_features=item_features, num_threads=num_threads).mean()
                return auc, prec, rec
            else:
                auc = auc_score(model, interactions, user_features=user_features, item_features=item_features, num_threads=num_threads).mean()
                prec = precision_at_k(model, interactions, k=k, user_features=user_features, item_features=item_features, num_threads=num_threads).mean()
                rec = recall_at_k(model, interactions, k=k, user_features=user_features, item_features=item_features, num_threads=num_threads).mean()
                return auc, prec, rec
        except Exception as e:
            raise

    def run(self, k=10, sample_users=5000, num_threads=4):
        try:
            with open(os.path.join(self.model_dir, "lightfm_model.pkl"), "rb") as f:
                model = pickle.load(f)
            interactions = sp.load_npz(os.path.join(self.model_dir, "interactions.npz"))
            user_features = sp.load_npz(os.path.join(self.model_dir, "user_features.npz"))
            item_features = sp.load_npz(os.path.join(self.model_dir, "item_features.npz"))

            logging.info("Starting evaluation...")
            auc, prec, rec = self.sampled_eval(model, interactions, user_features, item_features, k=k, sample_users=sample_users, num_threads=num_threads)

            logging.info(f"AUC: {auc:.4f}")
            logging.info(f"Precision@{k}: {prec:.4f}")
            logging.info(f"Recall@{k}: {rec:.4f}")

            print(f"AUC: {auc:.4f}")
            print(f"Precision@{k}: {prec:.4f}")
            print(f"Recall@{k}: {rec:.4f}")
        except Exception as e:
            raise AppException(e, sys) from e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample_users", type=int, default=5000)
    ap.add_argument("--num_threads", type=int, default=4)
    args = ap.parse_args()

    try:
        cfg = AppConfiguration()
        evaluator = ModelEvaluator(cfg)
        evaluator.run(k=args.k, sample_users=args.sample_users, num_threads=args.num_threads)
    except Exception as e:
        raise
