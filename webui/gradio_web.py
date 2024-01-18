import os
import sys
sys.path.append(os.getcwd())
import cv2
import argparse
import numpy as np
import gradio as gr
from PIL import Image
# from gradio_chat import Chat
from utils.conversation import conversation_lib


def parse_args():
    parser = argparse.ArgumentParser(description="u-LLaVA Chat")
    parser.add_argument("--llm_path", default="./exp/ullava/stage2_no_detach")
    parser.add_argument("--clip_processor", default="./model_zoo/clip-vit-large-patch14")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=-1, type=int)
    parser.add_argument("--conv_type", default='conv_sep2', type=str)
    parser.add_argument(
        "--dtype",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    return parser.parse_args()


args = parse_args()
# chat = Chat(args)


def plot_masks(gr_img, pred_masks, gr_gallery):
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        numpy_img = cv2.cvtColor(np.array(gr_img), cv2.COLOR_RGB2BGR)
        cv2_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)

        save_img = cv2_img.copy()
        save_img[pred_mask] = \
            (cv2_img * 0.5 + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5)[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        save_path = './vis_output/mask_img_{0}.jpg'.format(i)
        cv2.imwrite(save_path, save_img)

        gr_gallery = [Image.fromarray((pred_mask.astype(int) * 255).astype(np.uint8)),
                      cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB)]

    return save_path, gr_gallery


def upload_image(gr_img, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state
    else:
        chat_state = conversation_lib[args.conv_type].copy()
        return \
            gr_img, \
            gr.update(interactive=True, placeholder='Type and press Enter'), \
            gr.update(value="Start Chatting", interactive=False), \
            chat_state


def gradio_seg(user_message, gr_img, gr_mask, chat_state):
    # if len(user_message) == 0:
    #     return gr.update(interactive=True, placeholder='Input should not be empty!')
    if chat_state is not None:
        chat_state.messages = []

    if gr_img is not None:
        # llm_message, pred_masks = chat.seg(user_message, gr_img, chat_state)
        # gr_mask = pred_masks
        # print(llm_message)
        chat_state.messages = []

    return '', gr_mask, chat_state


def gradio_mask(gr_img, gr_mask, gr_gallery):

    save_path, gr_gallery = plot_masks(gr_img, gr_mask, gr_gallery)

    return gr_gallery,


def gradio_reset(chat_state):
    if chat_state is not None:
        chat_state.messages = []
    return \
        gr.update(value=None, interactive=True), \
        gr.update(placeholder='Please upload your image/video first', interactive=False), \
        gr.update(value="Upload & Start Chat", interactive=True), \
        gr.update(value=None), \
        gr.update(value=None), \
        chat_state


def init_demo():
    title_markdown = ("""
    # u-LLaVA: Unifying Multi-Modal Tasks via Large Language Model
    [[[Code](https://github.com/OPPOMKLab/u-LLaVA)] [[Model](https://huggingface.co/jinxu95/ullava)] | 📚 [[u-LLaVA](https://arxiv.org/pdf/2311.05348.pdf)]]
    """)

    tos_markdown = ("""
    ### Terms of use
    By using this service, users are required to agree to the following terms:
    The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
    Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
    For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
    """)

    learn_more_markdown = ("""
    ### License
    The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA.
    """)

    block_css = """

    #buttons button {
        min-width: min(120px,100%);
    }

    """

    text_box = gr.Textbox(label='User',
                          placeholder='Please upload your image first', interactive=False)

    with gr.Blocks(title="u-LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        gr.Markdown(title_markdown)
        # gr.Markdown(article)

        with gr.Row():
            with gr.Column(scale=3):
                models = ['ullava-7b', 'ullava-7b-lora']
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                image_box = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    [f"{cur_dir}/exp1.png", "Where does the dog sit?"],
                    [f"{cur_dir}/exp2.jpg", "Find the sun."],
                ], inputs=[image_box, text_box])

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                            label="Temperature", )
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P", )
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True,
                                                  label="Max output tokens", )

            with gr.Column(scale=8):
                with gr.Row():
                    mask_state = gr.State()
                    chat_state = gr.State()
                    gallery = gr.Gallery(label="Mask images", show_label=False, elem_id="gallery", columns=[2], rows=[1],
                                         object_fit="contain", height=550)

                    chat_bot = gr.Chatbot(elem_id="chatbot", label="u-LLaVA Chatbot", height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        text_box.render()
                    with gr.Column(scale=1, min_width=50):
                        upload_button = gr.Button(value="Send", variant="primary")
                        clear = gr.Button("Restart")

        upload_button.click(fn=gradio_seg,
                            inputs=[text_box, image_box, mask_state, chat_state],
                            outputs=[text_box, mask_state, chat_state]) \
            .then(fn=gradio_mask,
                  inputs=[image_box, mask_state, gallery],
                  outputs=[gallery])

        clear.click(fn=gradio_reset,
                    inputs=[chat_state],
                    outputs=[image_box, text_box, upload_button,
                             mask_state, gallery, chat_state],
                    queue=False)

    demo.launch(share=True, inbrowser=True)
    # demo.launch(server_name='0.0.0.0', server_port=6006, inbrowser=True)
    demo.queue(True)


if __name__ == "__main__":
    init_demo()
