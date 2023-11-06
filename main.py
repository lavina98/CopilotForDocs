from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
import re
from keybert import KeyBERT
import nltk
from nltk import WordNetLemmatizer, sent_tokenize, word_tokenize
from dotenv import load_dotenv
import os
import pinecone
from transformers import pipeline

def main():
    #nltk.download('wordnet')
    #nltk.download('punkt')
    load_dotenv()   
    api_key = os.getenv("PINECONE_API_KEY")
    pinecone.init(api_key=api_key, environment='us-west4-gcp-free')
    if 'honda-city-manual' not in pinecone.list_indexes():
        load_data()

    index = pinecone.Index("honda-city-manual")
    #load_data()
    question = "What does honda do with the data it acquires"
    keyword_model =  KeyBERT()
    keyword = keyword_model.extract_keywords(question, keyphrase_ngram_range=(1, 1), stop_words=None)
    #print(keyword)
    question_keyword = ' '.join([element[0] for element in keyword])
    
    model_name = "BAAI/bge-base-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    embedded_keys = model.embed_query(question_keyword)
    results_obj = index.query(queries=[embedded_keys], top_k=2, include_metadata=True)
    results = results_obj['results'][0]['matches']
    
    context= ''
    for result in results:
       context += result['metadata']['data'] + '.'
    
    qa_model_name = "deepset/roberta-base-squad2"
    nlp = pipeline('question-answering', model=qa_model_name, tokenizer=qa_model_name)
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    print(question)
    print(res)

def load_data():
    directory = os.path.dirname(__file__) 
    pdfPath = os.path.join(directory, 'HondaCityManual.pdf')
    reader = PdfReader(pdfPath)
    num_pages = len(reader.pages)
    pages = [reader.pages[i].extract_text() for i in range(0, 10)]
    cleaned_text_with_newlines = [re.sub(r"[^a-zA-Z0-9\s\.\?,!]", "", page) for page in pages]
    cleaned_text = [re.sub(r"\n", " ", page) for page in cleaned_text_with_newlines]
    #print(cleaned_text[1])
    lemmatizer = WordNetLemmatizer()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
        length_function = len,
        is_separator_regex=False
    )
    chunks = []
    for page in cleaned_text:
        chunks.extend(text_splitter.split_text(page))

    lemmatized_chunks = []
    for chunk in chunks:
        lemmatized_chunk = ' '. join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(chunk)])
        lemmatized_chunks.append(lemmatized_chunk)

    keyword_model =  KeyBERT()
    keywords = keyword_model.extract_keywords(lemmatized_chunks, keyphrase_ngram_range=(1, 1), stop_words=None)

    keys = []
    for arr in keywords:
        els = [element[0] for element in arr]
        els.sort()
        keys.append(' '.join(el for el in els))
    
    model_name = "BAAI/bge-base-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    embedded_keys = model.embed_documents(keys)
    
    key_embeddings =[]
    for i in range(len(embedded_keys)):
        key_embeddings.append((f'{keys[i]}', embedded_keys[i] , { 'data':chunks[i]}))
   

    load_dotenv()   
    api_key = os.getenv("PINECONE_API_KEY")
    pinecone.init(api_key=api_key, environment='us-west4-gcp-free')

    pinecone.create_index("honda-city-manual", metric="cosine", dimension=768)
    index = pinecone.Index("honda-city-manual")
    index.upsert(key_embeddings)
    

def get_unique_char_set(pages):
    unique_set = set()
    for page in pages:
        unique_set.update(set(page))
    return unique_set

if __name__ == "__main__":
    main()