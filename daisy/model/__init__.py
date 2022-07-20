from .EASERecommender import EASE
from .FMRecommender import FM
from .Item2VecRecommender import Item2Vec
from .KNNCFRecommender import ItemKNNCF
from .MFRecommender import MF
from .NeuMFRecommender import NeuMF
from .NFMRecommender import NFM
from .NGCFRecommender import NGCF
from .PopRecommender import MostPop
from .PureSVDRecommender import PureSVD
from .SLiMRecommender import SLiM
from .VAECFRecommender import VAECF

__all__ = [
    'EASE', 'FM', 'Item2Vec', 'ItemKNNCF', 'MF', 'NeuMF', 'NFM', 'NGCF', 'MostPop',
    'PureSVD', 'SLiM', 'VAECF',
]