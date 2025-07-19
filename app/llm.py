from groq import Groq
import streamlit as st

# api_key =  st.secrets["API_KEY"]
client = Groq(api_key='api_key')


system_prompt = '''You are an expert summarizer helping a user prepare input for a sentence embedding model with a strict 512-token input limit.
The user will always include the full **title** of the news article. Your job is to **rewrite or summarize the news article text only**, using no more than the available tokens provided. The goal is to preserve **all important meaning** from the article without exceeding the token budget.

You must:
- Leave the title unchanged
- Output **only the rewritten article text**
- Not include anything non-relevant stuff in your response

You will be provided Input details.
'''

def return_prompt(
    title: str,
    text: str,
    title_token_count: int,
    text_token_count: int,
    available_tokens_for_summarization: int,
    max_tokens: int = 512
):
    """
    Generates a prompt to summarize or rewrite the input news text to fit within
    the token limit of an embedding model, preserving key information.

    Args:
        title (str): The title of the news article.
        text (str): The original text of the news article.
        title_token_count (int): The number of tokens in the title.
        text_token_count (int): The number of tokens in the original text.
        available_tokens_for_summarization (int): The number of tokens available for summarizing the text.
        max_tokens (int): The maximum allowed total tokens for the title and text combined, default is 512.
    Returns:
        str: The formatted prompt for the summarization model.
    """
    prompt = f"""Input details:
    
    Title: {title}\n
    Title Token Count: {title_token_count}\n
    Original Text: {text}\n
    Original Text Token Count: {text_token_count}\n
    Max Allowed Total Tokens (Title + Text): {max_tokens}\n
    Available Tokens for Text Summary: {available_tokens_for_summarization}\n
    
    Now rewrite the article text only (not the title) within the available token budget. Your output should be clean, self-contained, and informative. 
    Do not add any information that is not present in the original news text or do not censor the text.
    Your output must be exactly text summary and nothing else. You are also not allowed to water-down or censor the text as the text may contain sensitive information related 
    to politics, race or religion. Your job is to just summarize the text and not censor/water-down the text in any way shape or form.
    The summary should be concise, informative, and maintain the original meaning.
    """
    return prompt

     
def run_groq_summary(prompt_text: str, combined_text: str, max_tokens: int, model="llama-3.3-70b-versatile"):
    '''
    Given a prompt and combined text, run the Groq API to summarize the provided text.

    Args:
        prompt_text (str): The prompt to guide the summarization.
        combined_text (str): The text to be summarized.
        max_tokens (int): The maximum number of tokens allowed in the output.
        model (str): The model to use for summarization, default is "llama-3.3-70b-versatile".
    Returns:
        str: The summarized text.
    '''
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content":system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            model=model,
            temperature=0.1,
            top_p=1,
            max_completion_tokens=max_tokens # this is the maximum number of tokens possible (title tokens + max_tokens = 512)
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        st.error(f"Error during summarization: {str(e)}")
        return combined_text # if api not working, pass full text and tokenizer will just truncate