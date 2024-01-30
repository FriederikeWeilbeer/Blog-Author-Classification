# Custom transformer to convert sparse matrix to dense array
from sklearn.base import TransformerMixin


class DenseTransformer(TransformerMixin):
    def transform(self, x, **transform_params):
        return x.toarray()

    def fit(self, x, y=None, **fit_params):
        return self

    def __str__(self):
        """
        String building method.
        """
        return "DenseTransformer"
