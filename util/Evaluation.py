from enum import Enum

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix


class ComparisonAttribute(Enum):
    ACCURACY = 'accuracy'
    RECALL = 'recall'
    PRECISION = 'precision'
    F1 = 'f1'
    ABSOLUTE = 'absolute'

    def __str__(self):
        return self.value


class Evaluation:

    def __init__(self, label_true, label_pred):
        self.label_true = label_true
        self.label_pred = label_pred
        self.tn, self.fp, self.fn, self.tp = self.__get_aggregated_confusion_matrix()
        self.accuracy = accuracy_score(label_true, label_pred)
        self.recall = recall_score(label_true, label_pred, average="macro")
        self.precision = precision_score(label_true, label_pred, average="macro")
        self.f1 = f1_score(label_true, label_pred, average="macro")
        self.comp_attr = ComparisonAttribute.ABSOLUTE

    def __get_aggregated_confusion_matrix(self):
        conf_matrix = confusion_matrix(self.label_true, self.label_pred)
        # Determine the number of unique classes
        num_classes = len(np.unique(self.label_true))

        print("Number Classes:", num_classes)

        # Aggregate metrics across all labels
        if num_classes > 2:
            fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
            fn = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
            tp = np.diag(conf_matrix)
            tn = conf_matrix.sum() - (fp + fn + tp)
            return tn, fp, fn, tp

        else:
            # Handle the case when there are only two labels
            return conf_matrix.ravel()

    def get_metrics(self):
        metrics_dict = {
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn,
            'tp': self.tp,
            'accuracy': self.accuracy,
            'recall': self.recall,
            'precision': self.precision,
            'f1': self.f1
        }
        return metrics_dict

    def get_correct_by_category(self):
        labels = np.unique(self.label_true)

        df = pd.DataFrame({'label_true': self.label_true, 'label_pred': self.label_pred})

        # Filter rows where both values are the same
        filtered_df = df[df['label_true'] == df['label_pred']]

        return filtered_df['label_true'].value_counts().to_dict()

    def __eq__(self, other):
        """
        Equality comparison method.
        """
        if isinstance(other, Evaluation):
            if self.comp_attr == ComparisonAttribute.ABSOLUTE:
                return all(getattr(self, attr.value) == getattr(other, attr.value) for attr in ComparisonAttribute if attr != ComparisonAttribute.ABSOLUTE)
            return getattr(self, self.comp_attr.value) == getattr(other, self.comp_attr.value)
        return False

    def __lt__(self, other):
        """
        Less than comparison method.
        """
        if isinstance(other, Evaluation):
            if self.comp_attr == ComparisonAttribute.ABSOLUTE:
                return all(getattr(self, attr.value) < getattr(other, attr.value) for attr in ComparisonAttribute if attr != ComparisonAttribute.ABSOLUTE)
            return getattr(self, self.comp_attr.value) < getattr(other, self.comp_attr.value)
        return NotImplemented

    def __le__(self, other):
        """
        Less than or equal to comparison method.
        """
        if isinstance(other, Evaluation):
            if self.comp_attr == ComparisonAttribute.ABSOLUTE:
                return all(getattr(self, attr.value) <= getattr(other, attr.value) for attr in ComparisonAttribute if attr != ComparisonAttribute.ABSOLUTE)
            return getattr(self, self.comp_attr.value) <= getattr(other, self.comp_attr.value)
        return NotImplemented

    def __gt__(self, other):
        """
        Greater than comparison method.
        """
        if isinstance(other, Evaluation):
            if self.comp_attr == ComparisonAttribute.ABSOLUTE:
                return all(getattr(self, attr.value) > getattr(other, attr.value) for attr in ComparisonAttribute if attr != ComparisonAttribute.ABSOLUTE)
            return getattr(self, self.comp_attr.value) > getattr(other, self.comp_attr.value)
        return NotImplemented

    def __ge__(self, other):
        """
        Greater than or equal to comparison method.
        """
        if isinstance(other, Evaluation):
            if self.comp_attr == ComparisonAttribute.ABSOLUTE:
                return all(getattr(self, attr.value) >= getattr(other, attr.value) for attr in ComparisonAttribute if attr != ComparisonAttribute.ABSOLUTE)
            return getattr(self, self.comp_attr.value) >= getattr(other, self.comp_attr.value)
        return NotImplemented

    def __str__(self):
        """
        String building method.
        """
        return (f'Accuracy: {self.accuracy} \nPrecision: {self.precision} \nRecall: {self.recall} \nF1: {self.f1}'
                f'\nTrue Positive: {self.tp} \nTrue Negative: {self.tn} \nFalse Positive: {self.fp} \nFalse Negative: {self.fn}')


class EvaluationResult:
    def __init__(self, evaluation: Evaluation, state):
        self.evaluation = evaluation
        self.state = state

    def __eq__(self, other):
        if isinstance(other, EvaluationResult):
            return self.evaluation == other.evaluation
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, EvaluationResult):
            return self.evaluation != other.evaluation
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, EvaluationResult):
            return self.evaluation < other.evaluation
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, EvaluationResult):
            return self.evaluation <= other.evaluation
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, EvaluationResult):
            return self.evaluation > other.evaluation
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, EvaluationResult):
            return self.evaluation >= other.evaluation
        return NotImplemented

    def __str__(self):
        """
        String building method.
        """
        return f'Evaluation: \n{self.evaluation} \nState: \n{self.state}'
