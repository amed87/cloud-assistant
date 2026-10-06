import re

def filterIntent(query:str)-> str:
    """Filter the Query to extract the intent from the user input. 
    The intent might be something like quantity", "location", "procedure", "troubleshooting" or "general"."""    
    query = query.lower().strip()
    
    if re.search(r"\b(wie (viele|viel|oft|lange|groß|hoch)|limit|max|anzahl|kapazität)\b", query):
        return "quantity"
        
    if re.search(r"\b(wo|wohin|woher|speicherort|pfad|ordner|verzeichnis|abgelegt)\b", query):
        return "location"
        
    if re.search(r"\b(wie (kann|funktionierte|richte|ändere|lösche)|anleitung|schritt)\b", query):
        return "procedure"
        
    if re.search(r"\b(warum|wieso|weshalb|fehler|error|problem|funktioniert nicht)\b", query):
        return "troubleshooting"
        
    return "general"
