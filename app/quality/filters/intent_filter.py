import re
from app.config.content_settings import content_config

def filterIntent(query: str) -> str:
    """Filter the Query to extract the intent from the user input. 
    The intent matches the metadata categories in the database."""   
    query_types = content_config.fragetyp_filter 
    query = query.lower().strip()

    # Filter according to the regex filters defined in the content config
    for intent, pattern in query_types.items():
        if re.search(pattern, query):
            return intent 
    
    # Fallback
    return content_config.fragetyp_fallback
