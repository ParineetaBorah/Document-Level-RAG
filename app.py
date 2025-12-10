from pdf_qa import MultimodalRAG
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from utils import convert_pdf_to_images
from loguru import logger
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from pydantic_models import PDFRequest, GenerateAnswerRequest

app=FastAPI()

CHROMA_DB_ROOT = "./chroma_db"  
PDF_ROOT = "./pdf"
PDF_IMAGE_ROOT="./pdf_images"
embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small")

@app.post("/process_pdf/")
def convert_to_images_endpoint(request: PDFRequest):
    try: 
        pdf_name = request.pdf_name
        pdf_path = os.path.join(PDF_ROOT, pdf_name)
        print(pdf_path)
        rag = MultimodalRAG(pdf_path=pdf_path)
        
        convert_pdf_to_images(pdf_path)
        logger.info({"message": f"PDF at {pdf_path} has been converted to images."})
        logger.info(f"Saved at {rag.pdf_images_dir}")

        elements = rag.partition_pdf()
        logger.info({"message": f"Partitioned PDF into {len(elements)} elements."})
        chunks = rag.create_chunks_by_title(elements)
        logger.info({"message": f"Created {len(chunks)} chunks from elements."})    
        langchain_documents = rag.create_document_langchain(chunks)
        logger.info({"message": f"Created {len(langchain_documents)} LangChain documents."})
        db = rag.create_vector_store(langchain_documents)
        logger.info({"message": "Vector store created successfully."})
        return {"status": "success", "collection_name": db, "message": "PDF processed and vector store created."}
    except Exception as e:
        logger.error(f"Error processing PDF at {pdf_path}: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/get_answer/")
def get_answer_endpoint(request: GenerateAnswerRequest):
    try:
        question = request.question
        collection_name = os.path.splitext(request.collection_name)[0]
        pdf_path = os.path.join(PDF_ROOT, request.pdf_name)
        db = Chroma(
            collection_name=collection_name,
            persist_directory=CHROMA_DB_ROOT,
            embedding_function=embedding_fn
        )

        rag = MultimodalRAG(pdf_path=pdf_path)

        retrieved_chunks = rag.retriever(query = question, vector_store=db)
        answer  = rag.generate_answer(retrieved_chunks, query=question)        
        logger.info({"message": f"Generated answer for question: {question}"})
        return {"status": "success", "answer": answer}
    except Exception as e:
        logger.error(f"Error generating answer for question '{question}': {e}")
        return {"status": "error", "message": str(e)}
    
@app.get("/pdf_images/{pdf_name}/page_{page_num}.png")
def get_page_image_endpoint(pdf_name: str, page_num: int):
    try:
        image_path = os.path.join(PDF_IMAGE_ROOT, pdf_name, f"page_{page_num}.png")
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")

        return FileResponse(image_path, media_type="image/png")
    except Exception as e:
        logger.error(f"Error retrieving image {image_path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


