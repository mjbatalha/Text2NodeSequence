# Text2NodeSequence
A PoC of an AI system capable of Text to Sequence of Nodes.

##  Instructions

1. Install libraries

```
pip install -r requirements.txt
```

2. Run

```
python main.py
```

## Load and run container

1. Load container

```
docker pull mjbatalha/text2node:1.0
```

2. Run container

```
docker run -it -p 7000:7000 mjbatalha/text2node:1.0
```

3. Run Gradio intereface

```
http://0.0.0.0:7000
```
n.b.: run local URL on a web browser.

## Structure

```
.
├── main.py
├── model.py
├── metrics.py
├── model_test.py
├── metrics_test.py
├── prompt_conf.yml
├── nodes.yml
├── examples.yml
├── Dockerfile
├── requirements.txt
├── README.md
└── LICENSE
```

## Evaluation metrics

The prompt/node-sequences provided in examples.yml are used as a validation set to obtain the following evaluation metrics. 

```
Precision   : 0.818
Recall      : 0.857
F1 Score    : 0.837
Exact Match : 0.429
Near Match  : 0.429
```
n.b.: Precision, Recal and F1 Score disregard node ordering and number of occurrences. Perhaps more interestingly, Exact and Near Match show proportion of examples where node generation either exactly matches or misses by only one node, respectively.


