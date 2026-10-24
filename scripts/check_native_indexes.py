"""检查原生索引确实返回k个邻居并找回自身；仅导入成功不算通过。"""
import json

import numpy as np
from annoy import AnnoyIndex
import hnswlib


def main():
    checks = []
    for dimension in (6, 128, 384):
        count = min(dimension, 16)
        vectors = np.eye(count, dimension, dtype=np.float32)
        annoy = AnnoyIndex(dimension, 'angular')
        for i, vector in enumerate(vectors):
            annoy.add_item(i, vector.tolist())
        annoy.build(10)
        hnsw = hnswlib.Index(space='cosine', dim=dimension)
        hnsw.init_index(max_elements=count, M=16, ef_construction=100)
        hnsw.add_items(vectors, np.arange(count))
        hnsw.set_ef(20)
        for i, vector in enumerate(vectors):
            neighbors, distances = annoy.get_nns_by_vector(vector.tolist(), 3, include_distances=True)
            if (len(neighbors) != 3 or len(set(neighbors)) != 3 or neighbors[0] != i
                    or not np.isfinite(distances).all()):
                raise ValueError(f'ANNOY原生查询失败：dimension={dimension}, id={i}, neighbors={neighbors}')
            neighbors, distances = hnsw.knn_query(vector.reshape(1, -1), k=3)
            if len(set(neighbors[0])) != 3 or neighbors[0][0] != i or not np.isfinite(distances).all():
                raise ValueError(f'HNSW原生查询失败：dimension={dimension}, id={i}')
        checks.append({'dimension': dimension, 'vectors': count, 'top_k': 3, 'all_self_queries_passed': True})
    print(json.dumps({'status': 'native_indexes_passed', 'checks': checks}, ensure_ascii=False))


if __name__ == '__main__':
    main()
