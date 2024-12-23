import torch
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer


class Text2NodeSeq():
    def __init__(
            self, 
            model_name: str = "unsloth/codegemma-7b-it",
            prompt_conf_path: str = "prompt_conf.yml",
            nodes_path: str = "nodes.yml",
            ):

        """
        Initializes the Text2NodeSeq class.

        Parameters
        ----------
        model_name : str, optional
            The name of the pre-trained model to use for generation. Defaults to "unsloth/codegemma-7b-it".
        prompt_conf_path : str, optional
            The path to the YAML file containing the prompt configuration. Defaults to "prompt_conf.yml".
        nodes_path : str, optional
            The path to the YAML file containing the node definitions. Defaults to "nodes.yml".
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        with open(prompt_conf_path, "r") as f:
            self.prompt_conf = yaml.safe_load(f)

        with open(nodes_path, "r") as f:
            self.nodes = yaml.safe_load(f)

    def prompt_preprocess(self, prompt: str) -> str:

        """
        Preprocesses a prompt by adding the node list and formatting text.
        
        Parameters
        ----------
        prompt : str
            The prompt to be preprocessed.
        
        Returns
        -------
        str
            The preprocessed prompt.
        """

        intro, prefix, suffix = self.prompt_conf.values()
        for node, action in self.nodes.items():
            intro = intro + f"\n{node} - {action}"
        prompt = intro + prefix + prompt + suffix

        return prompt

    def prompt_postprocess(self, out_string: str) -> str:

        """
        Postprocesses the output of the model by mapping the generated text to
        the corresponding node names.

        Parameters
        ----------
        out_string : str
            The output of the model.

        Returns
        -------
        str
            The postprocessed output as a list with the sequence of node names.
        """

        node_map = {k.lower(): f"[{k}]" for k in self.nodes.keys()}
        node_seq = out_string.split()
        pp_nodes = []
        for node in node_seq:
            node = node_map.get(node.lower())
            if node:
                pp_nodes.append(node)

        return pp_nodes

    def get_node_seq(self, prompt: str) -> list[str]:

        """
        Given a prompt, this function generates a sequence of node names that can 
        potentially fulfill the task described in the prompt. The function first 
        preprocesses the prompt by adding the node list and formatting text. It then 
        uses the model to generate a sequence of text based on the prompt. Finally, 
        it postprocesses the output of the model by mapping the generated text to 
        the corresponding node names. The function returns the sequence of node names 
        as a list of strings.

        Parameters
        ----------
        prompt : str
            The prompt to be processed.

        Returns
        -------
        list[str]
            The sequence of node names as a list of strings.
        """
        
        prompt = self.prompt_preprocess(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = inputs.shape[-1]
        outputs = self.model.generate(inputs, max_length=prompt_len + 100)
        out_string = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        node_seq = self.prompt_postprocess(out_string)

        return node_seq


if __name__ == "__main__":

    with open("examples.yml", "r") as f:
        examples = yaml.safe_load(f)

    model = Text2NodeSeq()

    for example in examples:

        prompt = example["prompt"]
        node_seq = model.get_node_seq(prompt)

        print(prompt)
        print("Expected:", example["nodes"])
        print("Generated:", node_seq)
        print("\n")

