# Import necessary modules from the transformers library
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#This makes conversation history list.
conversation_history = []

"""
During each interaction, you will pass your conversation history to the model along with 
your input so that it may also reference the previous conversation when generating the next answer.

You'll use the join() method in Python to do exactly that. (Initially, your history_string will be an 
empty string, which is okay, and will grow as the conversation goes on).
"""
history_string = "\n".join(conversation_history)

input_text ="hello, how are you doing?"

"""
Tokens in NLP are individual units or elements that text or sentences are divided into. Tokenization 
or vectorization is the process of converting tokens into numerical representations. In NLP tasks, you 
often use the encode_plus method from the tokenizer object to perform tokenization and vectorization. 
Let's encode the inputs (prompt & chat history) as tokens so that you may pass them to the model.
"""
inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
print(inputs)

#This attribute provides a mapping of pretrained models to their corresponding vocabulary files.
tokenizer.pretrained_vocab_files_map

outputs = model.generate(**inputs)
print(outputs)

"""
You may decode the output using tokenizer.decode(). This is known as "detokenization" or 
"reconstruction". It is the process of combining or merging individual tokens back into 
their original form, to reconstruct the original text or sentence.
"""
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(response)

#add both the input and response to conversation_history in plaintext.
conversation_history.append(input_text)
conversation_history.append(response)
print(conversation_history)

#Put everything in a loop to run whole conversation
while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)