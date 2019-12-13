try:
    from .main import main
except (ModuleNotFoundError, ImportError):
    from main import main
