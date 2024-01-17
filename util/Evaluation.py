from enum import Enum


class ComparisonAttribute(Enum):
    ACCURACY = 'accuracy'
    RECALL = 'recall'
    PRECISION = 'precision'
    F1 = 'f1'
    ABSOLUTE = 'absolute'

    def __str__(self):
        return self.value


class Evaluation:

    def __init__(self, accuracy, recall, precision, f1):
        self.accuracy = accuracy
        self.recall = recall
        self.precision = precision
        self.f1 = f1
        self.comp_attr = ComparisonAttribute.ABSOLUTE

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
        return f'Accuracy: {self.accuracy} \nPrecision: {self.precision} \nRecall: {self.recall} \nF1: {self.f1}'


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
