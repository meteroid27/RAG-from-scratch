def chunk_transcript(segments: list[dict], chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    chunks = []
    current_text = ""
    current_start = None

    for segment in segments:
        if current_start is None:
            current_start = segment["start"]
        
        current_text += " " + segment["text"]
        
        if len(current_text) >= chunk_size:
            chunks.append({
                "text": current_text.strip(),
                "start": current_start
            })
            
            current_text = current_text[-overlap:]
            current_start = segment["start"]  
    
    if current_text.strip():
        chunks.append({
            "text": current_text.strip(),
            "start": current_start
        })
    
    return chunks
