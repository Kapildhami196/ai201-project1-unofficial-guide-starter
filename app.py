import gradio as gr
from rag import answer_question


def handle_query(question):
    if not question.strip():
        return "Please enter a question.", ""

    result = answer_question(question)

    sources_text = "\n".join(f"• {src}" for src in result["sources"])

    return result["answer"], sources_text


with gr.Blocks(title="International Student Survival Guide") as demo:
    gr.Markdown("# 🌍 The International Student Survival Guide")
    gr.Markdown(
        "Ask questions about studying in the USA as an international student. "
        "Answers are based on your collected documents."
    )

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Your question",
                placeholder="Example: How do I build credit in the US without an SSN?",
                lines=2,
            )

            ask_button = gr.Button("Ask", variant="primary")

            answer_output = gr.Textbox(
                label="Answer",
                lines=12,
            )

            sources_output = gr.Textbox(
                label="Retrieved from",
                lines=5,
            )

    gr.Markdown("### Try these example questions:")

    gr.Examples(
        examples=[
            "What are the biggest mistakes international students make in their first semester?",
            "How can I build credit in the US without a SSN?",
            "What work can I legally do on an F-1 student visa?",
            "How hard is it to find a job after graduation on OPT?",
            "How do I deal with cultural shock and make friends in the US?",
        ],
        inputs=question_input,
    )

    ask_button.click(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output],
    )

    question_input.submit(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output],
    )


if __name__ == "__main__":
    demo.launch()
