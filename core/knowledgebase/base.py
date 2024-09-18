from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union

Data = Union[str, bytes]  # Text or binary data
Vector = List[float]

class VectorDatabaseInterface(ABC):
    @abstractmethod
    def connect(self) -> None:
        """
        Establish a connection to the database.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close the connection to the database.
        """
        pass

    @abstractmethod
    def insert(self, data: Data, vector: Vector, metadata: Optional[dict] = None) -> str:
        """
        Insert data into the database.
        """
        pass

    @abstractmethod
    def search(self, vector: Vector, k: int = 10, modality: Optional[str] = None) -> List[dict]:
        """
        Search for the k nearest neighbors of the query.
        The modality parameter can specify the type of data being searched (e.g., 'vector', 'text', 'image').
        """
        pass


