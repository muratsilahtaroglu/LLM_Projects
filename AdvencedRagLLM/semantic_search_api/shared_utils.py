from typing import Any, List, Tuple, Dict,Literal, Optional

embeding_models: Dict[str, Any]={}
"""
A dictionary to store embedding models, where the key is a string representing the model's name or identifier,
and the value can be the model instance, configuration, or any relevant data associated with the embedding model.
Initialized as an empty dictionary by default.
"""

#search_predictors:Dict[str,Dict[UUID, SemanticSearchApp]]={}
search_predictors={}