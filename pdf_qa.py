import json
from typing import List

import os

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv



load_dotenv()



class MultimodalRAG:
    def __init__(self, pdf_path:str):
        self.pdf_path = pdf_path
        self.pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.pdf_images_dir = f"./pdf_images/{self.pdf_name}/"
    
    def partition_pdf(self):
        elements = partition_pdf(filename=self.pdf_path,
                                 strategy = "hi_res",
                                 infer_table_structure=True,
                                 extract_image_block_types=["image"],
                                 extract_image_block_to_payload=True)
        
        print(f"Extracted {len(elements)} elements from the PDF.")
        return elements
    
    def create_chunks_by_title(self, elements):
        chunks = chunk_by_title(elements,
                                max_characters=3000,
                                new_after_n_chars=2400,
                                combine_text_under_n_chars=500)
        print(f"Chunked into {len(chunks)} sections based on titles.")
        return chunks
    
    def separate_chunk_contents(self, chunk):
        content_dict = {
            "text": chunk.text,
            "images":[],
            "tables":[],
            "types":["text"], 
            "page_number": []
            }
        
        if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
            for el in chunk.metadata.orig_elements:
                element_type = type(el).__name__
                if el.category == "Table":
                    table_html = getattr(el.metadata, 'text_as_html', el.text)
                    content_dict["types"].append("table")
                    content_dict["tables"].append(table_html)

                elif element_type == "Image":
                    if hasattr(el, "metadata") and hasattr(el.metadata, "image_base64"):
                        image_data = el.metadata.image_base64
                        content_dict["images"].append(image_data)
                        content_dict["types"].append("image")

        content_dict["page_number"] = list(set(self.pdf_images_dir+ f"page_{e.metadata.page_number}.png" for e in chunk.metadata.orig_elements))
        
        content_dict["types"] = list(set(content_dict["types"]))
        return content_dict
    
    def ai_summary(self, text, tables, images):

        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0)

            prompt = f'''You are creating a searchable description for a document chunk. 
            
            CONTENTS:
            TEXT: {text}
'''

            if tables:
                prompt+= "TABLE:\n"
                for i, table in enumerate(tables):
                    prompt+=f'''Table{i+1}:\n{table}\n\n'''

            prompt+='''Generate a comprehensive, searchable description from the text, tables and images provided that covers:
            - Key topics and concepts discussed
            - Questions that this conetnt could answer
            - Important data points from the tables
            - Any notable relationships or insights
            - Visual content analysis(charts, diagrams, patterns in images)
            - Alternative search terms users might use

            Make it detailed and searchable - prioritize findability over brevity.

            SEARCHABLE DESCRIPTION:
            '''

            message_content = [{"type":"text", "text": prompt}]

            for image_base64 in images:
                message_content.append({"type":"image_url", 
                                        "image_url": {"url":f"data:image/jpeg;base64,{image_base64}"}})
                
            message = HumanMessage(content=message_content)
            response = llm.invoke([message])

            return response.content
        
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"{text[300:]}..."
            
    def create_document_langchain(self, chunks):
        langchain_document = []
        for idx, chunk in enumerate(chunks):
            print(f"Processing {idx}/{len(chunks)} chunk...")
            content_dict = self.separate_chunk_contents(chunk)

            if content_dict["images"] or content_dict["tables"]:
                try: 
                    print(f"Generating AI summary for mixed content chunk ...")
                    content = self.ai_summary(content_dict["text"],
                                            content_dict["tables"],
                                            content_dict["images"])
                    
                    print(f"Successfully generated AI summary")
                
                except Exception as e:
                    content = content_dict["text"]
                    print(f"Falling back to text content due to error: {e}")

            
            else:
                print("No mixed content - using text only.")
                content = content_dict["text"]

            doc = Document(
                page_content=content,
                metadata={
                "original_content": json.dumps({
                    "text": content_dict["text"],
                    "tables": content_dict["tables"],
                    "images": content_dict["images"],
                    "page_numbers": content_dict["page_number"],
                })
                })
            
            langchain_document.append(doc)
        print(f"Processed {len(langchain_document)} chunks.")
        return langchain_document
    
    def create_vector_store(self, documents, persist_directory = "./chroma_db"):
        embedding_model = OpenAIEmbeddings(model = 'text-embedding-3-small')

        print(f"CReatig vector store in directory: {persist_directory} ...")
        collection_name = os.path.splitext(self.pdf_name)[0]
        vector_store = Chroma.from_documents(
            documents,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
            collection_metadata= {"hnsw:space":"cosine"}
        )
        print("__Finished storing to vector datastore.__")
        return collection_name
    
    def retriever(self, query, vector_store, k=5):
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        chunks = retriever.invoke(query)
        return chunks
    
    def get_page_links(self, chunks):
        page_links=[]
        for chunk in chunks:
            text_dict = json.loads(chunk.metadata["original_content"])
            page_links.extend(text_dict["page_numbers"])
        return list(set(page_links))
    
    def generate_answer(self, chunks, query):
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0)

            prompt = f'''Based on the following documents, generate a concise and accurate answer:\n\n
            
            CONTEXT:
            '''

            for i, chunk in enumerate(chunks):
                prompt+=f"--------------- Document {i+1} --------------\n"
                if "original_content" in chunk.metadata:
                    original_content = json.loads(chunk.metadata["original_content"])
                    prompt+=f"TEXT:\n{original_content['text']}\n\n"
                    if original_content["tables"]:
                        prompt+="TABLES:\n"
                        for j, table in enumerate(original_content["tables"]):
                            prompt+=f"Table {j+1}:\n{table}\n\n"

            prompt+="\n"

            prompt+=f'''Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

            ANSWER:'''

            message_content = [{"type":"text", "text": prompt}]

            for chunk in chunks:
                if "original_content" in chunk.metadata:
                    original_content = json.loads(chunk.metadata["original_content"])
                    for image_base64 in original_content["images"]:
                        message_content.append({"type":"image_url", 
                                                "image_url": {"url":f"data:image/jpeg;base64,{image_base64}"}})


            message = HumanMessage(content=message_content)
            response = llm.invoke([message])
            page_links = self.get_page_links(chunks)
            answer = {"final_answer": response.content,
                      "page_links": page_links}

            return answer
        
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Sorry, couldn't generate an answer due to an error."

                