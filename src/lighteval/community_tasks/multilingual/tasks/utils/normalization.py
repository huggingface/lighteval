def preprocess_thai_text(text):
    import pythainlp
    # Normalize Thai text
    text = pythainlp.util.normalize(text)
    
    # Remove any extra spaces
    text = " ".join(text.split())
    
    return text