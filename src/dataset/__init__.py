from src.dataset.huggingface import HuggingfaceDataset
from src.dataset.dailylife import DailylifeDataset
from src.dataset.toole import ToolEDataset
from src.dataset.multimedia import MultimediaDataset

load_dataset = {
    'huggingface': HuggingfaceDataset,
    'dailylife': DailylifeDataset,
    'toole': ToolEDataset,
    'multimedia': MultimediaDataset,
}
