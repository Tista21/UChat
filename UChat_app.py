import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# UI setup
st.title('UChat')
openai_api_key = st.sidebar.text_input('OpenAI API Key', type="password")
video_id = st.sidebar.text_input('Enter YouTube Video ID')
language_code = st.sidebar.text_input('Enter language code (e.g., "en")')

# Chain components outside to avoid redeclaration
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=openai_api_key)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def generate_response(input_text, video_id, language_code):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        return "No captions available for this video."

    chunks = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })  
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    return main_chain.invoke(input_text)

# Input form
with st.form('my_form'):
    input_text = st.text_area('Enter your question:', 'Can you summarize the video?')
    submitted = st.form_submit_button('Submit')

    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    elif submitted:
        if not video_id or not language_code:
            st.warning('Please provide both video ID and language code.', icon='⚠')
        else:
            result = generate_response(input_text, video_id, language_code)
            st.info(result)
