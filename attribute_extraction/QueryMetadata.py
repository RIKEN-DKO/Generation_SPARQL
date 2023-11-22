from SPARQLWrapper import SPARQLWrapper, CSV
import pandas as pd
from queries import *
from pandas import DataFrame

class QueryMetadata:

    def __init__(self, sparql_endpoint: str) -> None:
        super().__init__()
        self.sparql_endpoint = sparql_endpoint

    def get_all_metadata(self,  concept_list: dict = None, is_from_example: bool = False) -> DataFrame:
        sparql = SPARQLWrapper(self.sparql_endpoint)
        if concept_list is not None and len(concept_list) < 5:
            if is_from_example and len(concept_list) == 1:
                query = q3.replace("$$$", self.dict_to_sparql_values(concept_list))
            else:
                query = q2.replace("$$$", self.dict_to_sparql_values(concept_list))
        else:
            if concept_list is not None and len(concept_list) >= 5:
                csv_result_panda = pd.DataFrame({})
                for element in concept_list:
                    sublist = [element]
                    csv_result_panda = csv_result_panda._append(self.get_all_metadata(sublist, True), ignore_index=True)
                return csv_result_panda.drop_duplicates()
            else:
                query = q1
        print(query)

        sparql.setQuery(query)
        sparql.setReturnFormat(CSV)
        results = sparql.query().response
        csv_result_panda = pd.read_csv(results, sep=',', low_memory=False)
        return csv_result_panda.drop_duplicates()

    def dict_to_sparql_values(self, list: dict):
        values = ""
        for element in list:
           values += '<' + element + '> '
        return values