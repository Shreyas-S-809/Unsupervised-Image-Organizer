"""Serve the web interface and existing precomputed image artifacts."""
import argparse
import csv
import io
import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image
from plotly.offline import get_plotlyjs
import playground

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / 'app' / 'static'

@lru_cache(maxsize=1)
def load_data():
    with (ROOT / 'viz_data.csv').open(newline='', encoding='utf-8') as source:
        rows = [dict(x=float(r['x']), y=float(r['y']), z=float(r['z']),
                     kmeans_cluster=int(r['kmeans_cluster']),
                     dbscan_cluster=int(r['dbscan_cluster']), image_id=int(r['image_id']))
                for r in csv.DictReader(source)]
    return rows, np.load(ROOT / 'image_data.npy', mmap_mode='r', allow_pickle=False)

@lru_cache(maxsize=1)
def discovery_data():
    """Rank neighbors in CNN space and representatives in clustering (PCA) space."""
    rows, images = load_data()
    features = np.load(ROOT / 'app' / 'image_features.npy', allow_pickle=False)
    pca = np.load(ROOT / 'app' / 'pca_features.npy', allow_pickle=False)
    if len(features) != len(images) or len(pca) != len(images):
        raise ValueError('Feature artifacts must align with image indices')
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)
    scores = np.clip(normalized @ normalized.T, -1, 1)
    np.fill_diagonal(scores, -np.inf)
    # Zero-length embeddings have no defined cosine similarity.
    valid = norms[:, 0] > 0
    scores[:, ~valid] = -np.inf
    neighbors = []
    for image_id, values in enumerate(scores):
        ranked = np.argsort(-values, kind='stable')[:6] if valid[image_id] else []
        neighbors.append([{'image_id': int(i), 'score': float(values[i])}
                          for i in ranked if np.isfinite(values[i])])
    sheets = {}
    for method in ('kmeans_cluster', 'dbscan_cluster'):
        groups = []
        for cluster in sorted({r[method] for r in rows} - {-1}):
            ids = np.array([r['image_id'] for r in rows if r[method] == cluster])
            distances = np.linalg.norm(pca[ids] - pca[ids].mean(axis=0), axis=1)
            representatives = ids[np.argsort(distances, kind='stable')[:6]]
            groups.append({'cluster': cluster, 'count': len(ids),
                           'images': representatives.tolist()})
        sheets[method] = {'groups': groups, 'noiseCount': sum(r[method] == -1 for r in rows)}
    return {'neighbors': neighbors, 'sheets': sheets}

@lru_cache(maxsize=1)
def plotly_bundle():
    return get_plotlyjs().encode('utf-8')

class Handler(BaseHTTPRequestHandler):
    def send_content(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/playground/status':
            self.send_content(json.dumps(playground.status()).encode(), 'application/json')
        elif path == '/api/data':
            rows, images = load_data()
            self.send_content(json.dumps({'points': rows, 'imageCount': len(images),
                                          **discovery_data()}).encode(), 'application/json')
        elif path.startswith('/api/images/'):
            try:
                image_id = int(path.rsplit('/', 1)[-1])
                _, images = load_data()
                if not 0 <= image_id < len(images):
                    raise ValueError('Image ID out of range')
            except ValueError:
                self.send_error(404, 'Image not found')
                return
            output = io.BytesIO()
            Image.fromarray(images[image_id].astype('uint8')).save(output, format='PNG')
            self.send_content(output.getvalue(), 'image/png')
        elif path == '/plotly.min.js':
            self.send_content(plotly_bundle(), 'text/javascript; charset=utf-8')
        elif path in {'/', '/index.html', '/styles.css', '/app.js'}:
            name = 'index.html' if path == '/' else path[1:]
            kind = {'html': 'text/html', 'css': 'text/css', 'js': 'text/javascript'}[name.rsplit('.', 1)[-1]]
            self.send_content((STATIC / name).read_bytes(), kind + '; charset=utf-8')
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != '/api/playground/predict':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= playground.MAX_BYTES:
                self.send_content(json.dumps({'error': 'Choose an image smaller than 10 MB.'}).encode(), 'application/json', 413)
                return
            body = self.rfile.read(length)
            result = playground.predict(body)
            group = next(g for g in discovery_data()['sheets']['kmeans_cluster']['groups']
                         if g['cluster'] == result['cluster'])
            result.update(clusterSize=group['count'], representatives=group['images'])
            self.send_content(json.dumps(result).encode(), 'application/json')
        except ValueError as exc:
            self.send_content(json.dumps({'error': str(exc)}).encode(), 'application/json', 400)
        except playground.PlaygroundUnavailable as exc:
            self.send_content(json.dumps({'error': str(exc)}).encode(), 'application/json', 503)
        except Exception:
            self.log_error('Playground inference failed')
            self.send_content(json.dumps({'error': 'The image could not be analyzed. Check model setup and try again.'}).encode(), 'application/json', 500)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    load_data()
    print(f'Image Organizer is available at http://{args.host}:{args.port}', flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
