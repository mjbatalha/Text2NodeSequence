import yaml

from model import Text2NodeSeq


t2ns = Text2NodeSeq()

with open("nodes.yml", "r") as f:
    nodes = yaml.safe_load(f)


def test_get_node_seq_is_list():
    """
    Test to ensure that the get_node_seq function returns a list.
    """
    prompt = "Some prompt..."
    node_seq = t2ns.get_node_seq(prompt)
    assert isinstance(node_seq, list)


def test_get_node_seq_gibberish():
    """
    Test to ensure that the get_node_seq function returns an empty list when 
    given a prompt that makes no sense.
    """
    prompt = "Some gibberish... Blablablablablabla..."
    node_seq = t2ns.get_node_seq(prompt)
    assert len(node_seq) == 0


def test_get_node_seq_domain():
    """
    Test to ensure that the get_node_seq function returns a non-empty list of 
    valid nodes.
    """
    node_list = [f"[{k}]" for k in nodes.keys()]
    prompt = "Navigate to a new page after a delay of 3 seconds when the user clicks a button."
    node_seq = t2ns.get_node_seq(prompt)
    assert len(node_seq) > 0
    for node in node_seq:
        assert node in node_list

