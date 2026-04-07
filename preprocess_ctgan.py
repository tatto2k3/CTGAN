import os
import json
import pickle
import logging
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer


# =============================
# CONFIG (GLOBAL)
# =============================
DATA_PATH = "./datasets/data_JAR2020.csv"
OUTPUT_DIR = "./data/processed"
ARTIFACT_DIR = "./artifacts"

LABEL_COL = "misstate"
YEAR_COL  = "fyear"

TEMPORAL_SPLIT = True
KNN_NEIGHBORS = 5
RANDOM_STATE  = 42


# =============================
# LOGGER
# =============================
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


# =============================
# CORE CLASS
# =============================
class Preprocessor:
    def __init__(self):
        self.imputer = None

        self.raw_cols = [
            'act','ap','at','ceq','che','cogs','csho','dlc','dltis','dltt',
            'dp','ib','invt','ivao','ivst','lct','lt','ni','ppegt','pstk',
            're','rect','sale','sstk','txp','txt','xint','prcc_f'
        ]

    # =============================
    # LOAD
    # =============================
    def load_data(self):
        df = pd.read_csv(DATA_PATH)
        logging.info(f"Loaded data: {df.shape}")
        return df

    # =============================
    # CLEANING
    # =============================
    def clean_data(self, df):
        df = df.drop(columns=[c for c in ['gvkey', 'p_aaer'] if c in df.columns])
        df = df.drop_duplicates()

        logging.info(f"After cleaning: {df.shape}")
        return df

    # =============================
    # FEATURE GROUPING
    # =============================
    def get_feature_groups(self, df):
        ratio_cols = [
            col for col in df.columns
            if col not in self.raw_cols + [LABEL_COL, YEAR_COL]
        ]

        feature_cols = [
            col for col in df.columns
            if col not in [LABEL_COL, YEAR_COL]
        ]

        return feature_cols, ratio_cols

    # =============================
    # SPLIT (70/20/10)
    # =============================
    def split_data(self, df):
        if TEMPORAL_SPLIT and YEAR_COL in df.columns:
            df = df.sort_values(by=YEAR_COL)

            n = len(df)
            train_end = int(0.7 * n)
            val_end   = int(0.9 * n)

            train_df = df.iloc[:train_end].copy()
            val_df   = df.iloc[train_end:val_end].copy()
            test_df  = df.iloc[val_end:].copy()

            logging.info("Temporal split (70/20/10)")

        else:
            from sklearn.model_selection import train_test_split

            train_df, temp_df = train_test_split(
                df,
                test_size=0.3,
                stratify=df[LABEL_COL],
                random_state=RANDOM_STATE
            )

            val_df, test_df = train_test_split(
                temp_df,
                test_size=1/3,
                stratify=temp_df[LABEL_COL],
                random_state=RANDOM_STATE
            )

            logging.warning("Random split used")

        logging.info(f"Train: {train_df.shape}")
        logging.info(f"Val  : {val_df.shape}")
        logging.info(f"Test : {test_df.shape}")

        return train_df, val_df, test_df

    # =============================
    # FILL RAW (MEDIAN FROM TRAIN)
    # =============================
    def fill_raw(self, train_df, val_df, test_df):
        for col in self.raw_cols:
            if col in train_df.columns:
                median = train_df[col].median()
                train_df[col] = train_df[col].fillna(median)
                val_df[col]   = val_df[col].fillna(median)
                test_df[col]  = test_df[col].fillna(median)

        return train_df, val_df, test_df

    # =============================
    # KNN IMPUTER
    # =============================
    def fit_imputer(self, train_df, ratio_cols):
        if len(ratio_cols) > 0:
            self.imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
            self.imputer.fit(train_df[ratio_cols])

    def transform_imputer(self, df, ratio_cols):
        if self.imputer is not None:
            df[ratio_cols] = self.imputer.transform(df[ratio_cols])
        return df

    # =============================
    # SIGNED LOG
    # =============================
    @staticmethod
    def signed_log(x):
        return np.sign(x) * np.log1p(np.abs(x))

    def transform(self, df):
        for col in self.raw_cols:
            if col in df.columns:
                df[col] = self.signed_log(df[col])
        return df

    # =============================
    # SAVE
    # =============================
    def save(self, train_df, val_df, test_df):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
        val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
        test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

        with open(f"{ARTIFACT_DIR}/imputer.pkl", "wb") as f:
            pickle.dump(self.imputer, f)

        logging.info("Saved train/val/test")

    # =============================
    # RUN
    # =============================
    def run(self):
        df = self.load_data()
        df = self.clean_data(df)

        feature_cols, ratio_cols = self.get_feature_groups(df)

        train_df, val_df, test_df = self.split_data(df)

        # ===== NO LEAKAGE =====
        train_df, val_df, test_df = self.fill_raw(train_df, val_df, test_df)

        self.fit_imputer(train_df, ratio_cols)

        train_df = self.transform_imputer(train_df, ratio_cols)
        val_df   = self.transform_imputer(val_df, ratio_cols)
        test_df  = self.transform_imputer(test_df, ratio_cols)

        train_df = self.transform(train_df)
        val_df   = self.transform(val_df)
        test_df  = self.transform(test_df)

        self.save(train_df, val_df, test_df)

        logging.info("Preprocessing DONE")

        return train_df, val_df, test_df


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    setup_logger()

    processor = Preprocessor()
    train_df, val_df, test_df = processor.run()