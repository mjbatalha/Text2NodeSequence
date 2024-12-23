import yaml


def get_metrics(model, examples):

    """
    Computes metrics for a Text2NodeSeq model given a list of examples.

    Parameters
    ----------
    model : Text2NodeSeq
        The model to be evaluated.
    examples : list
        A list of examples, each containing a prompt, the expected node sequence, and the generated node sequence.

    Returns
    -------
    A dictionary with the following metrics:
        - precision
        - recall
        - f1_score
        - exact_match
        - near_match
    """
    
    TP, FP, FN, EM, NM = 0, 0, 0, 0, 0
    for example in examples:

        prompt = example["prompt"]
        nodes = example["nodes"]
        nodes_pred = model.get_node_seq(prompt)

        TP += len(set(nodes) & set(nodes_pred))
        FP += len(set(nodes_pred) - set(nodes))
        FN += len(set(nodes) - set(nodes_pred))
        EM += len([i for i, j in zip(nodes_pred, nodes) if i == j]) == len(nodes)
        NM += abs(len([i for i, j in zip(nodes_pred, nodes) if i == j]) - len(nodes)) == 1

    metrics = {
        "precision": TP / (TP + FP),
        "recall": TP / (TP + FN),
        "f1_score": 2 * TP / (2 * TP + FP + FN),
        "exact_match": EM / len(examples),
        "near_match": NM / len(examples)
    }

    return metrics


if __name__ == "__main__":

    from model import Text2NodeSeq

    with open("examples.yml", "r") as f:
        examples = yaml.safe_load(f)

    model = Text2NodeSeq()
    metrics = get_metrics(model, examples)
    
    for k, v in metrics.items():
        print(k, round(v, 3))

