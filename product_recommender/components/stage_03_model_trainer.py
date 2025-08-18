# product_recommender/components/train_model.py
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.data import Dataset
from product_recommender.logger.log import logging
from product_recommender.config.configuration import AppConfiguration
from product_recommender.exception.exception_handler import AppException


class ModelTrainer:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.config = app_config.get_model_training_config()
            self.cleaned_dir = self.config.cleaned_dir
            self.model_dir = self.config.model_dir
            os.makedirs(self.model_dir, exist_ok=True)
        except Exception as e:
            raise AppException(e, sys) from e

    def _sanitize_items(self, items_df: pd.DataFrame) -> pd.DataFrame:
        # categorical_features may be saved as string; convert using ast.literal_eval if needed
        import ast
        items = items_df.copy()
        if items["categorical_features"].dtype == object:
            def parse_cell(x):
                try:
                    if isinstance(x, list):
                        return x
                    if pd.isna(x):
                        return []
                    return ast.literal_eval(x)
                except Exception:
                    return []
            items["categorical_features"] = items["categorical_features"].apply(parse_cell)
        return items

    def build_and_train(self, no_components=64, learning_rate=0.05, epochs=30, loss="warp", num_threads=4):
        try:
            # load engineered artifacts
            events = pd.read_csv(os.path.join(self.cleaned_dir, "eng_event_df.csv"), parse_dates=["timestamp"])
            users = pd.read_csv(os.path.join(self.cleaned_dir, "eng_user_features_df.csv"))
            items = pd.read_csv(os.path.join(self.cleaned_dir, "eng_item_features_df.csv"))

            items = self._sanitize_items(items)

            # build feature name lists
            user_feature_names = [c for c in users.columns if c not in ("visitorid", "first_activity", "last_activity")]

            # gather token set for item features
            cat_tokens = set()
            for lst in items["categorical_features"]:
                if isinstance(lst, list):
                    for t in lst:
                        cat_tokens.add(f"tok_{int(t)}")

            item_feature_names = ["num"] + sorted(list(cat_tokens)) + \
                                 [f"catid_{int(c)}" for c in items["categoryid"].dropna().astype(int).unique()] + \
                                 [f"catpid_{int(p)}" for p in items["parentid"].dropna().astype(int).unique()] + ["available_1"]

            # Fit Dataset using all users/items present in engineered data
            dataset = Dataset()
            dataset.fit(users=users["visitorid"].unique(), items=items["itemid"].unique(),
                        user_features=user_feature_names, item_features=item_feature_names)

            # Build interactions (ensure items in events exist in items list)
            valid_items = set(items["itemid"].unique())
            events = events[events["itemid"].isin(valid_items)].copy()

            interactions, weights = dataset.build_interactions(
                (int(u), int(i), float(w)) for u, i, w in zip(events["visitorid"], events["itemid"], events["final_weight"])
            )

            # user features tuples (real-valued)
            def sanitize_num(x):
                try:
                    v = float(x)
                except Exception:
                    v = 0.0
                if np.isinf(v) or np.isnan(v):
                    v = 0.0
                return v

            user_tuples = []
            for _, r in users.iterrows():
                feats = {f: sanitize_num(r.get(f, 0.0)) for f in user_feature_names}
                user_tuples.append((int(r["visitorid"]), feats))
            user_features = dataset.build_user_features(user_tuples, normalize=False)

            # item features tuples
            item_tuples = []
            for _, r in items.iterrows():
                feats = {}
                feats["num"] = sanitize_num(r.get("avg_numeric_feature", 0.0))
                if int(r.get("available", 0)) == 1:
                    feats["available_1"] = 1.0
                if not pd.isna(r.get("categoryid")):
                    feats[f"catid_{int(r['categoryid'])}"] = 1.0
                if not pd.isna(r.get("parentid")):
                    feats[f"catpid_{int(r['parentid'])}"] = 1.0
                cats = r.get("categorical_features", [])
                if isinstance(cats, list):
                    for t in cats:
                        feats[f"tok_{int(t)}"] = 1.0
                item_tuples.append((int(r["itemid"]), feats))
            item_features = dataset.build_item_features(item_tuples, normalize=False)

            # Train model
            model = LightFM(loss=loss, no_components=no_components, learning_rate=learning_rate)
            model.fit(interactions, sample_weight=weights,
                      user_features=user_features, item_features=item_features,
                      epochs=epochs, num_threads=num_threads)

            # Save artifacts
            with open(os.path.join(self.model_dir, "lightfm_model.pkl"), "wb") as f:
                pickle.dump(model, f)

            with open(os.path.join(self.model_dir, "dataset.pkl"), "wb") as f:
                pickle.dump(dataset, f)

            sp.save_npz(os.path.join(self.model_dir, "interactions.npz"), interactions)
            sp.save_npz(os.path.join(self.model_dir, "user_features.npz"), user_features)
            sp.save_npz(os.path.join(self.model_dir, "item_features.npz"), item_features)

            # save item popularity (internal indices)
            item_pop = np.array(interactions.sum(axis=0)).ravel()
            np.save(os.path.join(self.model_dir, "item_popularity.npy"), item_pop)

            # Save mapping dicts from dataset.mapping()
            try:
                user_map, item_map, feature_map = dataset.mapping()
            except Exception:
                # fallback if mapping API is different
                user_map = getattr(dataset, "_user_id_mapping", {})
                item_map = getattr(dataset, "_item_id_mapping", {})
            with open(os.path.join(self.model_dir, "user_mapping.pkl"), "wb") as f:
                pickle.dump(user_map, f)
            with open(os.path.join(self.model_dir, "item_mapping.pkl"), "wb") as f:
                pickle.dump(item_map, f)

            logging.info("Model training complete and artifacts saved.")
        except Exception as e:
            raise AppException(e, sys) from e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_components", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--loss", type=str, default="warp")
    ap.add_argument("--num_threads", type=int, default=4)
    args = ap.parse_args()

    try:
        cfg = AppConfiguration()
        trainer = ModelTrainer(cfg)
        trainer.build_and_train(no_components=args.no_components,
                                learning_rate=args.learning_rate,
                                epochs=args.epochs,
                                loss=args.loss,
                                num_threads=args.num_threads)
    except Exception as e:
        raise
