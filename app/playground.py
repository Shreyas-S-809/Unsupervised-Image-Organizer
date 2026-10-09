"""Upload inference using the notebook's MobileNetV2 and a validated PCA rebuild."""
import csv
import hashlib
import io
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'app' / 'models'
PROJECTION = MODELS / 'projection.npz'
WEIGHTS = MODELS / 'mobilenet.weights.h5'
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
DATASET_IMAGE_SIZE = (32, 32)
INFERENCE_LOCK = threading.Lock()

class PlaygroundUnavailable(Exception):
    pass


def artifact_signature():
    digest = hashlib.sha256()
    for path in (ROOT / 'app/image_features.npy', ROOT / 'app/pca_features.npy', ROOT / 'viz_data.csv'):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def decode_image(body):
    if not body or len(body) > MAX_BYTES:
        raise ValueError('Choose an image smaller than 10 MB.')
    try:
        with Image.open(io.BytesIO(body)) as source:
            if source.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise ValueError('Choose a JPEG, PNG or WebP image.')
            if source.width * source.height > MAX_PIXELS:
                raise ValueError('Choose an image with no more than 20 million pixels.')
            source.load()
            oriented = ImageOps.exif_transpose(source)
            rgba = oriented.convert('RGBA')
            background = Image.new('RGBA', rgba.size, 'white')
            background.alpha_composite(rgba)
            return background.convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('This file could not be read as an image. Try a JPEG, PNG or WebP.') from exc


def tensorflow_model(weights):
    # Keep the weights cache inside the project. Imports stay lazy so Overview
    # remains available even if the optional inference setup has not been run.
    os.environ.setdefault('KERAS_HOME', str(MODELS / 'keras-cache'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    model = tf.keras.applications.MobileNetV2(weights=weights, include_top=False, pooling='avg')
    model.trainable = False
    return tf, model


def embedding(tf, model, image):
    pixels = np.asarray(image, dtype=np.float32)
    # The notebook embeds 32x32 CIFAR images enlarged to 224x224. Feeding
    # full-resolution uploads directly creates a different texture/detail
    # distribution, which can collapse nearest-center assignments to one group.
    # Area resampling matches the dataset resolution before the exact existing
    # bilinear enlargement. Original dataset images remain unchanged.
    if pixels.shape[:2] != DATASET_IMAGE_SIZE:
        pixels = tf.image.resize(pixels, DATASET_IMAGE_SIZE, method='area')
    resized = tf.image.resize(pixels, (224, 224))
    processed = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
    return model(processed[None, ...], training=False).numpy()[0]


def prepare():
    """Rebuild only upload artifacts; never rewrite original data or labels."""
    from sklearn.decomposition import PCA
    MODELS.mkdir(parents=True, exist_ok=True)
    features = np.load(ROOT / 'app/image_features.npy', allow_pickle=False)
    saved = np.load(ROOT / 'app/pca_features.npy', allow_pickle=False)
    with (ROOT / 'viz_data.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    labels = np.full(len(features), -1, dtype=int)
    for row in rows:
        labels[int(row['image_id'])] = int(row['kmeans_cluster'])
    if len(rows) != len(features) or np.any(labels < 0):
        raise ValueError('Image IDs and feature rows must align before preparing Playground.')
    # Use the notebook parameters, explicitly choosing its randomized solver.
    pca = PCA(n_components=saved.shape[1], svd_solver='randomized', random_state=42)
    pca.fit(features)
    projected = pca.transform(features)
    cluster_ids = np.unique(labels)
    centers = np.stack([projected[labels == c].mean(axis=0) for c in cluster_ids])
    old_centers = np.stack([saved[labels == c].mean(axis=0) for c in cluster_ids])
    new_assignment = cluster_ids[np.linalg.norm(projected[:, None] - centers, axis=2).argmin(axis=1)]
    old_assignment = cluster_ids[np.linalg.norm(saved[:, None] - old_centers, axis=2).argmin(axis=1)]
    # PCA axes may differ in sign/rotation across versions. Compare geometry
    # after orthogonal alignment, not raw coordinates or arbitrary axis signs.
    u, _, vt = np.linalg.svd(projected.T @ saved)
    relative_error = float(np.linalg.norm(projected @ (u @ vt) - saved) / np.linalg.norm(saved))
    agreement = float(np.mean(new_assignment == old_assignment))
    if agreement != 1.0 or relative_error > 0.02:
        raise ValueError(f'PCA rebuild did not preserve existing clusters (agreement={agreement:.3f}, error={relative_error:.4f}). Export the original fitted PCA instead.')
    print('PCA rebuilt: all existing nearest-center assignments preserved.', flush=True)
    tf, model = tensorflow_model('imagenet')
    images = np.load(ROOT / 'image_data.npy', mmap_mode='r', allow_pickle=False)
    checks = []
    for index in (0, len(images)//2, len(images)-1):
        actual = embedding(tf, model, Image.fromarray(images[index].astype('uint8')))
        expected = features[index]
        cosine = float(actual @ expected / (np.linalg.norm(actual) * np.linalg.norm(expected)))
        if cosine < 0.999:
            raise ValueError('MobileNetV2 features do not match the saved dataset preprocessing.')
        checks.append(cosine)
    model.save_weights(WEIGHTS)
    np.savez(PROJECTION, mean=pca.mean_, components=pca.components_, centers=centers,
             cluster_ids=cluster_ids, signature=artifact_signature())
    report = {'nearest_center_agreement': agreement, 'aligned_projection_relative_error': relative_error,
              'feature_cosine_checks': checks, 'method': 'Rebuilt PCA; centroids from existing KMeans memberships'}
    (MODELS / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Playground is ready. Start the app with: python app/app.py', flush=True)


def status():
    ready = PROJECTION.exists() and WEIGHTS.exists()
    return {'ready': ready, 'message': 'Ready to explore' if ready else 'Playground needs one-time model setup. Run python app/prepare_playground.py, then try again.'}


@lru_cache(maxsize=1)
def runtime():
    if not status()['ready']:
        raise PlaygroundUnavailable(status()['message'])
    with np.load(PROJECTION, allow_pickle=False) as data:
        state = {key: data[key].copy() for key in data.files}
    if str(state['signature']) != artifact_signature():
        raise PlaygroundUnavailable('The dataset changed. Run python app/prepare_playground.py again and restart the app.')
    try:
        tf, model = tensorflow_model(None)
        model.load_weights(WEIGHTS)
    except ImportError as exc:
        raise PlaygroundUnavailable('Install requirements.txt and restart the app to enable Playground.') from exc
    features = np.load(ROOT / 'app/image_features.npy', allow_pickle=False)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)
    return tf, model, state, normalized


def predict(body):
    image = decode_image(body)
    with INFERENCE_LOCK:
        tf, model, state, normalized = runtime()
        vector = embedding(tf, model, image)
    projected = (vector - state['mean']) @ state['components'].T
    distances = np.linalg.norm(state['centers'] - projected, axis=1)
    cluster = int(state['cluster_ids'][distances.argmin()])
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError('No usable visual features were found. Try another image.')
    scores = np.clip(normalized @ (vector / norm), -1, 1)
    matches = np.argsort(-scores, kind='stable')[:6]
    return {'cluster': cluster, 'neighbors': [{'image_id': int(i), 'score': float(scores[i])} for i in matches],
            'width': image.width, 'height': image.height}
