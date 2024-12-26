import gradio as gr
from model import Text2NodeSeq


model = Text2NodeSeq()


with gr.Blocks() as interface:

    """
    Setup Gradio web browser interface.
    """

    gr.Markdown("# Text to Sequence of Nodes")
    gr.Markdown("Enter a text prompt to generate a sequence of nodes based on your input.")

    with gr.Row():
        prompt_input = gr.Textbox(
            label="Text Prompt",
            placeholder="Enter your prompt here (e.g., Fetch user data and display it)..."
        )
    
    output = gr.Textbox(label="Generated Sequence", interactive=False)

    def on_submit(prompt):
        return model.get_node_seq(prompt)
    
    submit_button = gr.Button("Generate")
    submit_button.click(on_submit, inputs=prompt_input, outputs=output)


if __name__ == "__main__":

    interface.launch(server_name="0.0.0.0", server_port=7000)

