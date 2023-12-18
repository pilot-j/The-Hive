import streamlit as st
import torch

with open('stories.txt', 'r', encoding='utf-8') as f:
    text = f.read()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


def predict(model, input_text):
    context = torch.tensor([encode(input_text)], dtype=torch.long, device=device)
    return decode(model.generate(context, max_new_tokens=1000)[0].tolist())

loaded_model = torch.load('full_model.pt')
loaded_model.eval()  # Set the model to evaluation mode

def main():
    # Load model and weights
    # model_weights = load_model_and_weights()

    # Animated heading
    st.markdown("<h1 style='text-align: center; color: white;'>Ask Anything to Our Mini GPT</h1>", unsafe_allow_html=True)

    # Text input box for initial question
    user_input = st.text_input("Hii, I am mini GPT, How can I help you?")

    # Button to trigger GPT model prediction for the initial question
    if st.button("Ask GPT"):
        if user_input:
            # Display loading spinner while GPT model is processing
            with st.spinner("Getting response from GPT..."):
                # Get GPT model prediction
                gpt_output = predict(loaded_model, user_input)
                # gpt_output = "we still not integrated our model"
                # Display GPT model output
                st.success("Here's your response:")
                st.write(f"You: {user_input}")
                st.write(f"GPT: {gpt_output}")

    # About section
    st.sidebar.header("About")
    st.sidebar.write("This is a simple GPT web app using Streamlit.")
    st.sidebar.write("Built by Us -")
    st.sidebar.write("Snehanshu Mukherjee\n\nand\n\nSiddharth Mishra")
    # Logo and LinkedIn profiles
    st.sidebar.markdown(
        """
        [LinkedIn - Snehanshu Mukherjee](https://www.linkedin.com/in/snehanshu-mukherjee-71125622a/)  
        [LinkedIn - Siddharth Mishra](https://www.linkedin.com/in/iamsiddharthmishra/) 
        
          
        [GitHub - Snehanshu Mukherjee](https://github.com/pilot-j)      
        [GitHub - Siddharth Mishra](https://github.com/RustyGrackle)
        """
    )

if __name__ == "__main__":
    main()
