from app.exceptions.base import FaqBotError


class FAQLoadError(FaqBotError):
    """Error raised while loading an FAQ file."""