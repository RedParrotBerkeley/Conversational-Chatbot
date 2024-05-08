Conversational Chatbot using facebook/blenderbot-400M-distill
This project provides a basic implementation of a conversational chatbot using Facebook's blenderbot-400M-distill model, built with the transformers library by Hugging Face. The chatbot leverages state-of-the-art natural language processing (NLP) techniques to maintain a conversation, referencing previous inputs and responses in real-time.

Features
Pre-trained Model: Utilizes facebook/blenderbot-400M-distill, a lightweight and versatile conversational model.
Interactive Conversation: Continuously maintains conversation context through persistent history tracking.
Customizable Conversations: Easily customize conversation flow with your prompts and responses.
How to Use
Installation: Ensure you have the transformers library and PyTorch installed.

bash
pip install transformers torch

Run the Script:
Clone the repository to your local environment.
Execute the main Python file to start the chatbot.

bash
python chatbot.py

Chat with the Bot:
Type your input to receive a response based on the ongoing conversation history.
The conversation history will grow with each interaction.

Example Usage
bash
> Hello, how are you doing?
I'm good, thank you for asking! How about you?
> I'm fine, thanks for asking.
That's great to hear! What would you like to talk about today?


Dependencies
transformers: A comprehensive NLP library.
torch: PyTorch deep learning framework.
Feel free to contribute or customize the chatbot to fit your needs!
