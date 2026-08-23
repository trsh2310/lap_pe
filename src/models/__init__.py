def _optional_import(module_name, class_name):
    try:
        module = __import__(module_name, fromlist=[class_name])
    except (ModuleNotFoundError, TypeError):
        return None
    return getattr(module, class_name)


SASRecModel = _optional_import("src.models.sasrec", "SASRecModel")
SASRecMLP = _optional_import("src.models.sasrec_MLP", "SASRecMLP")
SASRecEinv = _optional_import("src.models.sasrec_einv", "SASRecEinv")
SASRecRoPE = _optional_import("src.models.sasrec_rope", "SASRecRoPE")
TiSASRec = _optional_import("src.models.tisasrec", "TiSASRec")

__all__ = [
    name
    for name, value in {
        "SASRecModel": SASRecModel,
        "SASRecMLP": SASRecMLP,
        "SASRecEinv": SASRecEinv,
        "SASRecRoPE": SASRecRoPE,
        "TiSASRec": TiSASRec,
    }.items()
    if value is not None
]
