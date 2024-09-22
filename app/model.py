from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Function to load the AI model (can replace GPT-2 with any other model)
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    return model, tokenizer

# Function to generate coding or API helper response
def generate_code_response(model_data, prompt, max_tokens):
    model, tokenizer = model_data
    # Encode the input prompt into tokens
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    # Generate the model's output
    outputs = model.generate(inputs, max_length=max_tokens, num_return_sequences=1)
    # Decode the output tokens to text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
