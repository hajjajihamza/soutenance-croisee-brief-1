from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

"""Limiter les heures de réseaux sociaux au temps d'écran."""
class SocialMediaCapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['Social_Media_Hours'] = X['Social_Media_Hours'].clip(upper=X['Screen_Time_Hours'])

        return X


""" Transformateur sklearn : plafonnement avec IQR """
class IQRCapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        Q1, Q3 = np.quantile(X, [0.25, 0.75], axis=0)

        IQR = Q3 - Q1

        self.lower_ = Q1 - 1.5 * IQR
        self.upper_ = Q3 + 1.5 * IQR

        return self

    def transform(self, X):
        return np.clip(X, a_min=self.lower_, a_max=self.upper_)