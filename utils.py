import io
import fitz
import os


def convert_pdf_to_images(pdf_path: str):
    images = []
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(f"pdf_images/{pdf_name}/", exist_ok=True)
    print(f"Starting to convert pdf: {pdf_name}")
    with fitz.open(pdf_path) as doc:
        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                images.append(img_bytes)

                with open(f"pdf_images/{pdf_name}/page_{page_num + 1}.png", "wb") as img_file:
                    img_file.write(img_bytes)
        except Exception as e:
            print(f"Error converting page {page_num} to image: {e}")