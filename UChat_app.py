from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
st.title('UChat')
openai_api_key = st.sidebar.text_input('OpenAI API Key')
video_id = st.sidebar.text_input('Enter video ID')
Langu = st.sidebar.text_input('Enter language code')
def generate_response(input_text):
  video_id = "video_id"
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["Langu"])

    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")
  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text
parallel_chain = RunnableParallel({
  'context': retriever | RunnableLambda(format_docs),
  'question': RunnablePassthrough()
})  
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser
text = input_text 
  st.info(main_chain.invoke(text))
with st.form('my_form'):
  text = st.text_area('Enter text:', 'Can you summarize the video')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)
