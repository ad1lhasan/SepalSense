import gradio as gr
import pickle

# Define the prediction function (loads the model from the pickle file)
def predict_iris_class(sepal_length, sepal_width, petal_length, petal_width):
    # Load the model
    with open("model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
        
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = loaded_model.predict(input_data)[0]
    
    return prediction

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_iris_class,  # Correct function reference
    inputs=[
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)")
    ],
    outputs=gr.Textbox(label="Predicted Iris Class")  # Output to display prediction
)

# Launch the interface
iface.launch()
