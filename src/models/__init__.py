def _optional_import(module_name, class_name):
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ModuleNotFoundError:
        return None
    except TypeError:
        return None
    return getattr(module, class_name)


PopularRandom = _optional_import("src.models.popular_random", "PopularRandom")
UserKNNModel = _optional_import("src.models.user_knn", "UserKNNModel")
ItemKNNModel = _optional_import("src.models.item_knn", "ItemKNNModel")
SeqKNN = _optional_import("src.models.seq_knn", "SeqKNN")
EASE_R = _optional_import("src.models.ease_r", "EASE_R")
BPR_MF = _optional_import("src.models.bpr_mf", "BPR_MF")
ALSMF_sparse = _optional_import("src.models.als_mf", "ALSMF_sparse")
UltraGCN = _optional_import("src.models.ultragcn", "UltraGCN")
MFSGD = _optional_import("src.models.sgd_mf", "MFSGD")
LightGCN = _optional_import("src.models.lightgcn", "LightGCN")
RandomModel = _optional_import("src.models.random", "RandomModel")
PureSVDModel = _optional_import("src.models.pure_svd", "PureSVDModel")
GASATF = _optional_import("src.models.gasatf", "GASATF")
ALS_Implicit = _optional_import("src.models.als_implicit", "ALS_Implicit")
SASRecModel = _optional_import("src.models.sasrec", "SASRecModel")
JointSASRecUltraGCN = _optional_import("src.models.joint", "JointSASRecUltraGCN")
JointItems = _optional_import("src.models.i_joint", "JointItems")
SASRecItem = _optional_import("src.models.sasrec_I", "SASRecItem")
SASRecEinv = _optional_import("src.models.sasrec_einv", "SASRecEinv")
SASRecInter = _optional_import("src.models.sasrec_inter", "SASRecInter")
SASRecMLP = _optional_import("src.models.sasrec_MLP", "SASRecMLP")
SASRecRoPE = _optional_import("src.models.sasrec_rope", "SASRecRoPE")
SASRecRoPELapRaw = _optional_import("src.models.sasrec_rope_lap_raw", "SASRecRoPELapRaw")
SASRecRoPELapProjection = _optional_import("src.models.sasrec_rope_lap_projection", "SASRecRoPELapProjection")
SASRecRoPELapV = _optional_import("src.models.sasrec_rope_lap_projection", "SASRecRoPELapV")
SASRecRoPELapKV = _optional_import("src.models.sasrec_rope_lap_projection", "SASRecRoPELapKV")
SASRecRoPELapQK = _optional_import("src.models.sasrec_rope_lap_projection", "SASRecRoPELapQK")
SASRecLapAttentionBias = _optional_import("src.models.sasrec_lap_attention_bias", "SASRecLapAttentionBias")
TiSASRec = _optional_import("src.models.tisasrec", "TiSASRec")

__all__ = [
    name
    for name, value in {
        "PopularRandom": PopularRandom,
        "UserKNNModel": UserKNNModel,
        "ItemKNNModel": ItemKNNModel,
        "SeqKNN": SeqKNN,
        "EASE_R": EASE_R,
        "BPR_MF": BPR_MF,
        "ALSMF_sparse": ALSMF_sparse,
        "UltraGCN": UltraGCN,
        "MFSGD": MFSGD,
        "LightGCN": LightGCN,
        "RandomModel": RandomModel,
        "PureSVDModel": PureSVDModel,
        "GASATF": GASATF,
        "ALS_Implicit": ALS_Implicit,
        "SASRecModel": SASRecModel,
        "JointSASRecUltraGCN": JointSASRecUltraGCN,
        "JointItems": JointItems,
        "SASRecItem": SASRecItem,
        "SASRecEinv": SASRecEinv,
        "SASRecInter": SASRecInter,
        "SASRecMLP": SASRecMLP,
        "SASRecRoPE": SASRecRoPE,
        "SASRecRoPELapRaw": SASRecRoPELapRaw,
        "SASRecRoPELapProjection": SASRecRoPELapProjection,
        "SASRecRoPELapV": SASRecRoPELapV,
        "SASRecRoPELapKV": SASRecRoPELapKV,
        "SASRecRoPELapQK": SASRecRoPELapQK,
        "SASRecLapAttentionBias": SASRecLapAttentionBias,
        "TiSASRec": TiSASRec,
    }.items()
    if value is not None
]
