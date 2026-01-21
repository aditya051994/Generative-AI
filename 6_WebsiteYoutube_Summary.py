import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Prompt
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Streamlit config
st.set_page_config(page_title="LangChain: Summarize Text from YT or Website")
st.title("LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Sidebar input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")

# URL input
generic_url = st.text_input("Enter URL")

# Button action
if st.button("Summarize"):
    if not groq_api_key.strip():
        st.error("Please enter Groq API Key")
    elif not generic_url.strip():
        st.error("Please enter a URL")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Summarizing..."):

                # ✅ Create LLM ONLY HERE
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    groq_api_key=groq_api_key
                )

                # Load data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False
                    )
                    docs = loader.load()
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        }
                    )
                    docs = loader.load()

                # Summarization chain
                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                output_summary = chain.run(docs)

                st.success("Summary Generated")
                st.write(output_summary)

        except Exception as e:
            st.exception(e)
