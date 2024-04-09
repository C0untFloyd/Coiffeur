import time
import gradio as gr
import sys
import torch

version = "Coiffeur 0.1 - a HairClip V2 UI Fork"

def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def on_clicked_start(inputimg, txtdesc, refimg, radiosel, coltext, colpick):
    from hairmain import hairstyle_editing, align_faces

    align_faces(inputimg, "./test_images/src_img/")
    if refimg is not None:
        refimg = align_faces(refimg, "./test_images/ref_img/")

    if radiosel == "From Ref Image":
        color_cond = refimg
    elif radiosel == "Text description":
        #color_cond = f"{coltext}"
        color_cond = f"{coltext} hair"
    elif radiosel == "RGB Colors":
        color_cond = hex_to_rgb(colpick)
    else:
        color_cond = None
    

    _, finalimg = hairstyle_editing(inputimg, txtdesc, refimg, color_cond)
    return finalimg

def create_version_html() -> str:
    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    versions_html = f"""
python: <span title="{sys.version}">{python_version}</span>
•
torch: {getattr(torch, '__long_version__',torch.__version__)}
•
gradio: {gr.__version__}
"""
    return versions_html

def run():
    run_server = True
    mycss = """
        span {color: var(--block-info-text-color)}
        #fixedheight {
            max-height: 238.4px;
            overflow-y: auto !important;
        }
        .image-container.svelte-1l6wqyv {height: 100%}

    """

    while run_server:
        with gr.Blocks(title=f'{version}', css=mycss) as ui:
            with gr.Row(variant='compact'):
                    gr.Markdown(f"### [{version}](https://github.com/C0untFloyd/HairCLIPv2_UI)")
                    gr.HTML(create_version_html(), elem_id="versions")
            with gr.Row():
                with gr.Column():
                    inputimg = gr.Image(label='Input Image', sources=["upload"], type="filepath", interactive=True)
                    radiosel = gr.Radio(["Input image color", "From Ref Image", "Text description", "RGB Color"], label="Target Hair color", value="Input color", interactive=True)
                with gr.Column():
                    txtdesc = gr.Textbox(label="Enter hairstyle", placeholder="bowl cut hairstyle, quiff hairstyle", interactive=True)
                    refimg = gr.Image(label='Ref Image with Hairstyle', sources=["upload"], type="filepath", interactive=True)
            with gr.Row():
                coltext = gr.Textbox(label="Enter hair color", placeholder="red, blonde, brown, black, gray", interactive=True)
                colpick = gr.ColorPicker(label="color", interactive=True)
                gr.Markdown(" ")
                bt_start = gr.Button("Start")
            with gr.Row():
                finalimg = gr.Image(label='Resulting Image', interactive=False)
        
            bt_start.click(fn=on_clicked_start,  inputs=[inputimg, txtdesc, refimg, radiosel, coltext, colpick],outputs=[finalimg])


        restart_server = False
        try:
            ui.queue().launch(inbrowser=True, prevent_thread_lock=True, show_error=True)
        except Exception as e:
            print(f'Exception {e} when launching Gradio Server!')
            restart_server = True
            run_server = False
        try:
            while restart_server == False:
                time.sleep(1.0)

        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        ui.close()


if __name__ == '__main__':
    run()
