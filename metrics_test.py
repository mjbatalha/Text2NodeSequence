import yaml

from metrics import get_metrics
from model import Text2NodeSeq


t2ns = Text2NodeSeq()

with open("examples.yml", "r") as f:
    examples = yaml.safe_load(f)


def test_get_metrics_is_dict():
    """
    Test to ensure that the get_metrics function returns a dictionary.
    """
    metrics_dict = get_metrics(t2ns, examples)
    assert isinstance(metrics_dict, dict)


def test_get_metrics_is_float():
    """
    Test to ensure that the values returned by the get_metrics function are floats.
    """    
    metrics_dict = get_metrics(t2ns, examples)
    for v in metrics_dict.values():
        assert isinstance(v, float)


def test_get_metrics_domain():
    """
    Test to ensure that the values returned by the get_metrics function
    are between 0 and 1, inclusive.
    """
    metrics_dict = get_metrics(t2ns, examples)
    for v in metrics_dict.values():
        assert 0 <= v <= 1

