import logging
from abc import ABC, abstractclassmethod

from datasets import DatasetDict
from transformers import AutoTokenizer
from ingest_data import ingest_data

logger = logging.getLogger(__name__)