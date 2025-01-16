# algorithms/file_handler.py
import PyPDF2
from docx import Document
import io
import chardet
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def detect_encoding(binary_content):
    """Détecte l'encodage du fichier"""
    try:
        result = chardet.detect(binary_content)
        return result['encoding'] or 'utf-8'
    except Exception as e:
        logger.error(f"Erreur lors de la détection de l'encodage: {str(e)}")
        return 'utf-8'

def extract_text_from_pdf(file_content):
    """Extrait le texte d'un fichier PDF"""
    try:
        if not file_content:
            raise ValueError("Le contenu du fichier PDF est vide")
            
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        if len(pdf_reader.pages) == 0:
            raise ValueError("Le PDF ne contient aucune page")
            
        text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
                
        return ' '.join(text).strip()
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du texte du PDF: {str(e)}")
        raise Exception(f"Erreur lors de l'extraction du texte du PDF: {str(e)}")

def extract_text_from_docx(file_content):
    """Extrait le texte d'un fichier DOCX"""
    try:
        if not file_content:
            raise ValueError("Le contenu du fichier DOCX est vide")
            
        doc = Document(io.BytesIO(file_content))
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text.append(paragraph.text)
        
        if not text:
            raise ValueError("Le document DOCX ne contient aucun texte")
            
        return '\n'.join(text).strip()
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du texte du DOCX: {str(e)}")
        raise Exception(f"Erreur lors de l'extraction du texte du DOCX: {str(e)}")

def extract_text_from_file(file):
    """Extrait le texte d'un fichier en fonction de son type"""
    try:
        if not file:
            raise ValueError("Aucun fichier fourni")
            
        filename = file.filename.lower()
        file_content = file.read()
        
        if not file_content:
            raise ValueError(f"Le fichier {filename} est vide")
            
        # Reset the file pointer for potential future reads
        file.seek(0)
        
        if filename.endswith('.txt'):
            encoding = detect_encoding(file_content)
            text = file_content.decode(encoding).strip()
            if not text:
                raise ValueError(f"Le fichier texte {filename} ne contient aucun texte")
            return text
            
        elif filename.endswith('.pdf'):
            return extract_text_from_pdf(file_content)
            
        elif filename.endswith(('.doc', '.docx')):
            return extract_text_from_docx(file_content)
            
        else:
            raise ValueError(f"Format de fichier non supporté: {filename}")
            
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du texte de {file.filename}: {str(e)}")
        raise Exception(f"Erreur lors de l'extraction du texte: {str(e)}")