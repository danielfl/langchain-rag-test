from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader


class DocumentLoader:
    def __init__(self, url):
        self.loader = YoutubeLoader.from_youtube_url(url)

    def load(self):
        print("Document loader is loading documents...")
        return self.loader.load()


class TextSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap)

    def split(self, documents):
        print("Text splitter is splitting documents...")
        chunks = []
        for document in documents:
            chunks += self.text_splitter.create_documents(
                    [document.page_content],
                    [document.metadata])
        return chunks


class VectorStore:
    def __init__(self, data_path="./chroma_store", model_name="all-MiniLM-L6-v2", collection_name="mycol"):
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.client = chromadb.PersistentClient(path=data_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"},
        )

    def store(self, chunks):
        document_ids = list(map(lambda tup: f"id{tup[0]}", enumerate(chunks)))
        docs = [document.page_content for document in chunks]
        metas = [document.metadata for document in chunks]
        self.collection.add(documents=docs, metadatas=metas, ids=document_ids)


class LLMCaller:
    def __init__(self, model_path, n_gpu_layers, n_batch, n_threads, max_tokens, n_ctx):
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(
            model_path=model_path,
            callback_manager=self.callback_manager,
            input={"temperature": 0.75, "max_length": 4096, "top_p": 1},
            verbose=True,
            seed=0,
            use_mlock=True,
            streaming=True,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            max_tokens=max_tokens,
            n_threads=n_threads,
            n_batch=n_batch
        )
        self.template = PromptTemplate(
            input_variables=['context', 'question'],
            template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
        )

    def call(self, question, context):
        print(f"Question: {question}\n")
        llm_chain = LLMChain(prompt=self.template, llm=self.llm)
        result = llm_chain.run({"context": context, "question": question})
        print(result)
        print("\n#########################################\n")


loader = DocumentLoader("https://www.youtube.com/watch?v=8fEEbKJoNbU")
documents = loader.load()

splitter = TextSplitter(1000, 50)
chunks = splitter.split(documents)

store = VectorStore("chroma_fosdem2024_data/", "all-MiniLM-L6-v2", "fosdem_2024")
store.store(chunks)

llm_caller = LLMCaller('models/llama-2-13b-chat.Q5_K_M.gguf',
                       n_gpu_layers=32,
                       n_batch=16,
                       n_threads=8,
                       max_tokens=512,
                       n_ctx=4096)

questions = [
    "What is the e/acc movement?",
    "What is effective accelerationism?",
    "What is Kardashev scale?",
    "What energy sources could provide the needed energy?",
    "What is the difference between e/acc and effective altruism?",
    "What can you say about black holes?",
]

for question in questions:
    # get embeddings from a text query
    docs = store.collection.query(query_texts=[question], n_results=5)
    # create a context
    context = "\n\n".join(doc for doc in docs['documents'][0])
    # ask the question and context
    llm_caller.call(question, context)
