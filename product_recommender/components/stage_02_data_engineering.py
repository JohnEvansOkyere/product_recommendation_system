# product_recommender/components/data_engineering.py
import os
import sys
import re
import numpy as np
import pandas as pd
from product_recommender.logger.log import logging
from product_recommender.config.configuration import AppConfiguration
from product_recommender.exception.exception_handler import AppException


class DataEngineer:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.config = app_config.get_data_engineering_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _parse_value_field(value: str):
        # returns (list_of_numeric, list_of_tokens)
        if not isinstance(value, str):
            return [], []
        nums = []
        toks = []
        for part in value.strip().split():
            if part.startswith("n"):
                try:
                    nums.append(float(part[1:].replace(",", "")))
                except Exception:
                    continue
            else:
                if re.fullmatch(r"\d+", part):
                    toks.append(int(part))
        return nums, toks

    def build_item_features(self, item_props: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
        try:
            df = item_props.copy()
            df["parsed"] = df["value"].apply(self._parse_value_field)
            df["numeric_features"] = df["parsed"].apply(lambda x: x[0])
            df["categorical_features"] = df["parsed"].apply(lambda x: x[1])

            # available
            avail_mask = df["property_id"].astype(str).str.lower() == "available"
            if avail_mask.sum() > 0:
                avail = df.loc[avail_mask].assign(avl=lambda d: (d["value"].astype(str) == "1").astype(int))
                avail = avail.groupby("itemid")["avl"].max().rename("available")
            else:
                avail = pd.Series(dtype=int)

            # categoryid from property
            cat_mask = df["property_id"].astype(str).str.lower() == "categoryid"
            if cat_mask.sum() > 0:
                direct_cat = df.loc[cat_mask].assign(categoryid=lambda d: pd.to_numeric(d["value"], errors="coerce")).groupby("itemid")["categoryid"].max()
            else:
                direct_cat = pd.Series(dtype=float)

            # aggregate numeric & tokens
            agg = df.groupby("itemid").agg(
                avg_numeric_feature=("numeric_features", lambda lists: np.nanmean([x for l in lists for x in l]) if any(len(l) for l in lists) else 0.0),
                categorical_features=("categorical_features", lambda lists: list(sorted(set([x for l in lists for x in l]))))
            ).reset_index()

            # merge avail & direct_cat
            if not avail.empty:
                agg = agg.merge(avail.reset_index(), on="itemid", how="left")
            else:
                agg["available"] = 0
            if not direct_cat.empty:
                agg = agg.merge(direct_cat.reset_index(), on="itemid", how="left")
                agg = agg.rename(columns={"categoryid": "categoryid"})
            else:
                agg["categoryid"] = -1

            # merge parent info from categories if exists
            cats = categories.copy()
            if "categoryid" in cats.columns and "parentid" in cats.columns:
                cats = cats.rename(columns={"categoryid": "categoryid", "parentid": "parentid"})
                agg = agg.merge(cats[["categoryid", "parentid"]], on="categoryid", how="left")
            else:
                agg["parentid"] = -1

            agg["avg_numeric_feature"] = agg["avg_numeric_feature"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            agg["categoryid"] = agg["categoryid"].fillna(-1).astype(int)
            agg["parentid"] = agg["parentid"].fillna(-1).astype(int)
            agg["available"] = agg["available"].fillna(0).astype(int)

            return agg[["itemid", "avg_numeric_feature", "categorical_features", "categoryid", "available", "parentid"]]

        except Exception as e:
            raise

    def build_user_features(self, events: pd.DataFrame) -> pd.DataFrame:
        try:
            evt = events.copy().sort_values("timestamp")
            grp = evt.groupby("visitorid")
            agg = grp.agg(
                total_events=("itemid", "count"),
                unique_items=("itemid", "nunique"),
                first_activity=("timestamp", "min"),
                last_activity=("timestamp", "max"),
                views=("event", lambda s: (s == "view").sum()),
                addtocart=("event", lambda s: (s == "addtocart").sum()),
                transaction=("event", lambda s: (s == "transaction").sum())
            ).reset_index()

            agg["activity_duration_days"] = (agg["last_activity"] - agg["first_activity"]).dt.days.clip(lower=0).fillna(0)
            agg["activity_duration_days"] = agg["activity_duration_days"].replace(0, 1)
            agg["events_per_day"] = agg["total_events"] / agg["activity_duration_days"]
            agg["avg_events_per_item"] = agg["total_events"] / agg["unique_items"].replace(0, 1)
            agg["view_to_cart_ratio"] = np.where(agg["views"] > 0, agg["addtocart"] / agg["views"], 0.0)
            agg["cart_to_purchase_ratio"] = np.where(agg["addtocart"] > 0, agg["transaction"] / agg["addtocart"], 0.0)
            agg["view_to_purchase_ratio"] = np.where(agg["views"] > 0, agg["transaction"] / agg["views"], 0.0)
            agg["engagement_score"] = agg["views"] + 2 * agg["addtocart"] + 6 * agg["transaction"]
            agg = agg.fillna(0)
            return agg
        except Exception as e:
            raise

    def build_weighted_events(self, events: pd.DataFrame, half_life_days: int = 14) -> pd.DataFrame:
        try:
            EVENT_BASE_WEIGHT = {"view": 1.0, "addtocart": 3.0, "transaction": 6.0}
            evt = events.copy()
            max_ts = evt["timestamp"].max()
            evt["days_ago"] = (max_ts - evt["timestamp"]).dt.days.clip(lower=0)
            evt["recency_weight"] = 1.0 / (1.0 + (evt["days_ago"] / float(half_life_days)))
            evt["base_weight"] = evt["event"].map(EVENT_BASE_WEIGHT).astype(float)
            evt["final_weight"] = evt["base_weight"] * evt["recency_weight"]
            return evt[["timestamp", "visitorid", "event", "itemid", "final_weight", "days_ago", "recency_weight"]]
        except Exception as e:
            raise

    def run(self):
        try:
            self._ensure_dir(self.config.cleaned_dir)

            # load cleaned files created by preprocessing
            events = pd.read_csv(os.path.join(self.config.cleaned_dir, "events_clean.csv"), parse_dates=["timestamp"])
            item_props = pd.read_csv(os.path.join(self.config.cleaned_dir, "item_props_clean.csv"))
            categories = pd.read_csv(os.path.join(self.config.cleaned_dir, "category_clean.csv"))

            logging.info("Building engineered events (weights/recency)")
            eng_events = self.build_weighted_events(events, half_life_days=self.config.half_life_days if hasattr(self.config, "half_life_days") else 14)

            logging.info("Building user features")
            eng_users = self.build_user_features(events)

            logging.info("Building item features")
            eng_items = self.build_item_features(item_props, categories)

            # save engineered artifacts
            eng_events.to_csv(os.path.join(self.config.cleaned_dir, "eng_event_df.csv"), index=False)
            eng_users.to_csv(os.path.join(self.config.cleaned_dir, "eng_user_features_df.csv"), index=False)
            eng_items.to_csv(os.path.join(self.config.cleaned_dir, "eng_item_features_df.csv"), index=False)

            logging.info("Feature engineering saved to cleaned artifacts")
        except Exception as e:
            raise AppException(e, sys) from e


if __name__ == "__main__":
    try:
        cfg = AppConfiguration()
        eng = DataEngineer(cfg)
        logging.info("Starting data engineering")
        eng.run()
        logging.info("Data engineering finished successfully")
    except Exception as err:
        raise
