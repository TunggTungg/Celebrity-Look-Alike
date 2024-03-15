from elasticsearch import Elasticsearch

class DataBase():
    def __init__(self, ip_address="http://localhost:9200", re_init=False):
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
    def re_init(self):
        self.es.indices.create(index="face_recognition", body=self.mapping)

    def add_index(self):
        self.re_init()
        import numpy as np
        loaded = np.load('all_fearture.npz')
        identity = loaded["identity"]
        embedding = loaded["embedding"]
        index = 0
        for i in range(len(identity)):
            doc = {"title_vector": embedding[i][0], "title_name": identity[i]}
            self.es.index(index="face_recognition", id=index, body=doc)
            index = index + 1
     
    def search(self, targets):
        results = []
        for target in targets:
            self.query["query"]["script_score"]["script"]["params"]["queryVector"] = list(target) 
            result = self.es.search(index="face_recognition", body=self.query)["hits"]["hits"][0]
            results.append(result["_source"]["title_name"])
        return results

if __name__ == "__main__":
    db = DataBase()
    db.add_index()
