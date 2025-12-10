from pydantic import BaseModel

class PDFRequest(BaseModel):
    pdf_name: str

class GenerateAnswerRequest(BaseModel):
    question: str 
    collection_name: str 
    pdf_name: str