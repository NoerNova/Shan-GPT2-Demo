import gradio as gr
from gpt2 import generate_text, GENERATE_EXAMPLES

gpt_generate = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Input text"),
        gr.Dropdown(
            [
                "sample_outputs",
                "greedy_search",
                "beem_search",
                "top_k_search",
                "top_p_search",
            ],
            label="Search method",
            value="sample_outputs",
        ),
    ],
    outputs=gr.Textbox(label="Generated text"),
    examples=GENERATE_EXAMPLES,
    title="GPT-2 Text generator Demo",
    description="Generate text using GPT-2.",
    allow_flagging="never",
)

with gr.Blocks() as demo:
    gpt_generate.render()

demo.launch()
