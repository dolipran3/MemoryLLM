import json
from datasets import load_dataset
import gradio as gr

custom_css = """
.message.user {
    background-color: #ff8c00 !important;
}
"""

ds = load_dataset("bowen-upenn/PersonaMem-v2")


# print(ds["benchmark_text"].unique("pref_type"))


def historyIdUser(nameSplit, idUser):
    dsFilterById = ds[nameSplit].filter(lambda example: example["persona_id"] == idUser)
    return dsFilterById


def loadConversation(dsFiltered, index=0):
    if 0 <= index < len(dsFiltered):
        conversation = json.loads(dsFiltered["related_conversation_snippet"][index])
        return conversation, index
    return [], index


def nextConversation(dsFiltered, currentIndex):
    maxIndex = len(dsFiltered["related_conversation_snippet"]) - 1
    newIndex = min(currentIndex + 1, maxIndex)
    conversation, _ = loadConversation(dsFiltered, newIndex)
    return (
        conversation,
        newIndex,
        dsFiltered["topic_query"][newIndex],
        dsFiltered["preference"][newIndex],
        dsFiltered["topic_preference"][newIndex],
        dsFiltered["conversation_scenario"][newIndex],
        dsFiltered["sensitive_info"][newIndex],
        dsFiltered["pref_type"][newIndex],
    )


def previousConversation(dsFiltered, currentIndex):
    newIndex = max(currentIndex - 1, 0)
    conversation, _ = loadConversation(dsFiltered, newIndex)
    return (
        conversation,
        newIndex,
        dsFiltered["topic_query"][newIndex],
        dsFiltered["preference"][newIndex],
        dsFiltered["topic_preference"][newIndex],
        dsFiltered["conversation_scenario"][newIndex],
        dsFiltered["sensitive_info"][newIndex],
        dsFiltered["pref_type"][newIndex],
    )


def changeUser(nameSplit, idUser):
    dsFiltered = historyIdUser(nameSplit, idUser)
    conversation, _ = loadConversation(dsFiltered, 0)
    return (
        conversation,
        0,
        dsFiltered,
        dsFiltered["topic_query"][0],
        dsFiltered["preference"][0],
        dsFiltered["topic_preference"][0],
        dsFiltered["conversation_scenario"][0],
        dsFiltered["sensitive_info"][0],
        dsFiltered["pref_type"][0],
    )


if __name__ == "__main__":
    nameSplit = "benchmark_text"
    uniqueIds = ds[nameSplit].unique("persona_id")
    dsFiltered = historyIdUser(nameSplit=nameSplit, idUser=uniqueIds[0])
    conversation = json.loads(dsFiltered["related_conversation_snippet"][0])

    with gr.Blocks() as demo:
        dsFilteredState = gr.State(dsFiltered)
        currentIndex = gr.State(0)

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(conversation, height=800)

                with gr.Row():
                    btnPrev = gr.Button("<- Previous")
                    btnNext = gr.Button("Next ->")

            with gr.Column(scale=1):
                with gr.Row():
                    dropdown = gr.Dropdown(
                        choices=uniqueIds,
                        value=uniqueIds[0],
                        label="Sélectionner un Persona ID",
                        scale=29,
                    )
                    btnHelp = gr.Button("?", scale=1, size="sm")

                infoTextHelp = gr.Textbox(
                    label="Information",
                    value="Sélectionnez un ID de persona pour voir ses conversations",
                    interactive=False,
                    visible=False,
                )
                topicText = gr.Textbox(
                    label="Topic",
                    value=dsFilteredState.value["topic_query"][0],
                    interactive=False,
                )
                preferenceText = gr.Textbox(
                    label="Preference",
                    value=dsFilteredState.value["preference"][0],
                    interactive=False,
                )
                topicPrefText = gr.Textbox(
                    label="Topic Preference",
                    value=dsFilteredState.value["topic_preference"][0],
                    interactive=False,
                )
                scenarioText = gr.Textbox(
                    label="Conversation Scenario",
                    value=dsFilteredState.value["conversation_scenario"][0],
                    interactive=False,
                )
                sensitiveText = gr.Textbox(
                    label="Sensitive Info",
                    value=dsFilteredState.value["sensitive_info"][0],
                    interactive=False,
                )
                prefType = gr.Textbox(
                    label="Pref Type",
                    value=dsFilteredState.value["pref_type"][0],
                    interactive=False,
                )

        btnPrev.click(
            fn=lambda idx, ds_f: previousConversation(ds_f, idx),
            inputs=[currentIndex, dsFilteredState],
            outputs=[
                chatbot,
                currentIndex,
                topicText,
                preferenceText,
                topicPrefText,
                scenarioText,
                sensitiveText,
                prefType,
            ],
        )

        btnNext.click(
            fn=lambda idx, ds_f: nextConversation(ds_f, idx),
            inputs=[currentIndex, dsFilteredState],
            outputs=[
                chatbot,
                currentIndex,
                topicText,
                preferenceText,
                topicPrefText,
                scenarioText,
                sensitiveText,
                prefType,
            ],
        )

        btnHelp.click(
            fn=lambda: gr.update(visible=True),
            outputs=[infoTextHelp],
        )

        dropdown.change(
            fn=lambda id_user: changeUser(nameSplit, id_user),
            inputs=[dropdown],
            outputs=[
                chatbot,
                currentIndex,
                dsFilteredState,
                topicText,
                preferenceText,
                topicPrefText,
                scenarioText,
                sensitiveText,
                prefType,
            ],
        )

    demo.launch(css=custom_css)
