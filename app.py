from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_text(prompt, max_length=100, num_return_sequences=1, num_beams=5, repetition_penalty=2.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        attention_mask=inputs["attention_mask"]
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    response_text = generate_text(prompt, max_length=max_length)
    return jsonify({'generated_text': response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

