import time
import gradio as gr

def on_clicked_start(inputimg, txtdesc, refimg):
    from hairmain import hairstyle_editing

    _, finalimg = hairstyle_editing(inputimg, txtdesc, refimg)
    return finalimg

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
        with gr.Blocks(title=f'HairCLIP UI', css=mycss) as ui:
            with gr.Row():
                with gr.Column():
                    inputimg = gr.Image(label='Input Image', interactive=True)
                with gr.Column():
                    txtdesc = gr.Textbox(label="Enter hairstyle or hair color", placeholder="bowl cut hairstyle, blue hair, quiff hairstyle", interactive=True)
                    refimg = gr.Image(label='Ref Image with Hairstyle', interactive=True)
            with gr.Row():
                bt_start = gr.Button("Start")
            with gr.Row():
                finalimg = gr.Image(label='Resulting Image', interactive=False)
        
            bt_start.click(fn=on_clicked_start,  inputs=[inputimg, txtdesc, refimg],outputs=[finalimg])


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
