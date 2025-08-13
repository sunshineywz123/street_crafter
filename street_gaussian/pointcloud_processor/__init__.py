from .waymo_processor import WaymoPointCloudProcessor
from .pandaset_processor import PandasetPointCloudProcessor
from .sensetime_processor import SensetimePointCloudProcessor
from .base_processor import BasePointCloudProcessor
from street_gaussian.config import cfg

PointCloudProcessorType = {
    "Waymo": WaymoPointCloudProcessor,
    "Pandaset": PandasetPointCloudProcessor,
    "Sensetime": SensetimePointCloudProcessor
}


def getPointCloudProcessor() -> BasePointCloudProcessor:
    return PointCloudProcessorType[cfg.data.type]()
