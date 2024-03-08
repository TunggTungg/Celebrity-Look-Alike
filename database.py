from elasticsearch import Elasticsearch
import numpy as np

class DataBase():
    def __init__(self, ip_address="http://localhost:9200"):
        self.es = Elasticsearch(ip_address)
        self.mapping = {
        "mappings": {
            "properties": {
                "title_vector":{
                    "type": "dense_vector",
                    "dims": 512
                },
                "title_name": {"type": "keyword"}
                }
            }
        }
        self.query = {
            "size": 5,
            "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.queryVector, 'title_vector') + 1.0",
                            # "source": "1 / (1 + l2norm(params.queryVector, 'title_vector'))", #euclidean distance
                            "params": {
                                "queryVector": []
                            }
                        }
                    }
            }
        }  
    def search(self, target):
        self.query["query"]["script_score"]["script"]["params"]["queryVector"] = list(target) 
        result = self.es.search(index="face_recognition", body=self.query)["hits"]["hits"][0]
        # result = self.es.search(index="face_recognition", body=self.query)
        # for i in result["hits"]["hits"]:
        #     candidate_name = i["_source"]["title_name"]
        #     candidate_score = i["_score"]
        #     print(candidate_name, ": ", candidate_score)
        res = result["_source"]["title_name"]
        # print(result["_score"])
        return res
        