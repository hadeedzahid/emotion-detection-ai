import gradio as gr
from transformers import pipeline

# Load the pre-trained emotion detection model
# Using j-hartmann/emotion-english-distilroberta-base which is great for general emotion classification
model_path = "j-hartmann/emotion-english-distilroberta-base"
emotion_classifier = pipeline("text-classification", model=model_path, top_k=None)

def predict_emotion(text):
    """
    Predicts the emotion of the input text using the Hugging Face pipeline.
    """
    if not text.strip():
        return {}
    
    # The pipeline returns a list of lists when top_k=None
    predictions = emotion_classifier(text)
    
    # Format the predictions into a dictionary for Gradio's Label component
    # predictions[0] contains a list of dicts like {'label': 'joy', 'score': 0.9}
    emotion_scores = {pred['label'].capitalize(): float(pred['score']) for pred in predictions[0]}
    
    return emotion_scores

# Context Notes to display on the UI
context_notes = """
### Use Case Context: Why Emotion Detection Matters
- **Customer Service**: Automatically route frustrated customers to human agents or prioritize urgent complaints.
- **Healthcare**: Analyze patient feedback or therapy transcripts to monitor mental well-being and emotional progress.
- **Social Media Analysis**: Understand public sentiment around brand launches, political events, or trending topics in real-time.
"""

# Build the Gradio interface
with gr.Blocks(title="Emotion Detection AI") as demo:
    gr.Markdown("# Project 1: Emotion Detection using AI")
    gr.Markdown("Enter some text below, and the AI will analyze its emotional state (e.g., Joy, Anger, Sadness, Neutral, etc.).")
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Input Text", 
                placeholder="Type a sentence here... (e.g., 'I am absolutely thrilled about the new update!')",
                lines=5
            )
            submit_btn = gr.Button("Analyze Emotion", variant="primary")
        
        with gr.Column(scale=1):
            label_output = gr.Label(label="Predicted Emotions", num_top_classes=7)
            
    gr.Markdown("---")
    gr.Markdown(context_notes)
    
    # Define the interaction
    submit_btn.click(fn=predict_emotion, inputs=text_input, outputs=label_output)
    # Also trigger on pressing enter in the textbox
    text_input.submit(fn=predict_emotion, inputs=text_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
