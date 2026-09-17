"""Helsinki-only binary crop verification; recorded views are inference-only.

Frozen ImageNet EfficientNet features feed a logistic-regression head. Synthetic
validation shares source scenery and measures separation, not city transfer.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
DATASET = LAB / 'dataset_v5'
PATCH_SIZE = 128
CONTEXT = 1.25


def crop_bounds(box):
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    half = max(4.0, box[2] - box[0], box[3] - box[1]) * CONTEXT / 2
    return [center_x - half, center_y - half, center_x + half, center_y + half]


def crop(image, box):
    left, top, right, bottom = crop_bounds(box)
    height, width = image.shape[:2]
    left, top = max(0, int(np.floor(left))), max(0, int(np.floor(top)))
    right, bottom = min(width, int(np.ceil(right))), min(height, int(np.ceil(bottom)))
    if right <= left or bottom <= top:
        raise ValueError('crop does not intersect the view')
    patch_image = image[top:bottom, left:right]
    side = max(patch_image.shape[:2])
    padded = cv2.copyMakeBorder(patch_image, 0, side - patch_image.shape[0],
                                0, side - patch_image.shape[1], cv2.BORDER_REFLECT_101)
    return cv2.resize(padded, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)


def view_box(box, region, shape):
    height, width = shape[:2]
    left, top, right, bottom = region
    return [(box[0] - left) * width / (right - left),
            (box[1] - top) * height / (bottom - top),
            (box[2] - left) * width / (right - left),
            (box[3] - top) * height / (bottom - top)]


def clear(box, truth):
    return all(min(box[2], other[2]) <= max(box[0], other[0]) or
               min(box[3], other[3]) <= max(box[1], other[1]) for other in truth)


def read_truth(path, shape):
    from exp03_classical_ml import read_labels
    from dtos import OBJECT_CLASSES

    height, width = shape[:2]
    labels = DATASET / 'labels' / path.parent.name / (path.stem + '.txt')
    if not labels.is_file():
        raise FileNotFoundError(f'Missing labels; cannot treat as background: {labels}')
    return [{'object_id': OBJECT_CLASSES[index],
             'bbox': [center_x - box_width / 2, center_y - box_height / 2,
                      center_x + box_width / 2, center_y + box_height / 2]}
            for index, center_x, center_y, box_width, box_height in read_labels(labels, width, height)]


class Encoder:
    def __init__(self, pretrained=True):
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(4)
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        self.model.classifier = torch.nn.Identity()
        self.model.eval().requires_grad_(False).to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]

    @torch.inference_mode()
    def features(self, patches):
        if not len(patches):
            return np.empty((0, 1280), dtype=np.float32)
        result = []
        for start in range(0, len(patches), 64):
            images = np.asarray(patches[start:start + 64])[..., ::-1].copy()
            padded = np.zeros((64, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
            padded[:len(images)] = images
            tensor = torch.from_numpy(padded).permute(0, 3, 1, 2).to(self.device).float() / 255
            result.append(self.model((tensor - self.mean) / self.std)[:len(images)].cpu().numpy())
        return np.concatenate(result)


def mine(paths, detector, seed):
    from solution import iou

    generator = np.random.default_rng(seed)
    patches, labels, kinds, groups, scenes = [], [], [], [], []
    for image_index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read {path}')
        truth = read_truth(path, image.shape)
        boxes = [item['bbox'] for item in truth]
        height, width = image.shape[:2]
        predictions = detector(image, (0, 0, width, height))
        scenes.append({'image': str(path), 'truth': truth, 'predictions': predictions})

        def add(box, label, kind, group):
            patches.append(crop(image, box))
            labels.append(label)
            kinds.append(kind)
            groups.append(group)

        for item in truth:
            add(item['bbox'], 1, 'label_positive', item['object_id'])
        hard_count = 0
        for prediction in sorted(predictions, key=lambda item: -item['confidence']):
            box = prediction['bbox']
            if clear(crop_bounds(box), boxes):
                if hard_count < 16:
                    add(box, 0, 'hard_negative', 'background')
                    hard_count += 1
            elif boxes:
                best = max(truth, key=lambda item: iou(box, item['bbox']))
                if iou(box, best['bbox']) >= 0.5:
                    add(box, 1, 'detector_positive', best['object_id'])
        negative_count = 0
        for _ in range(120):
            if negative_count >= 12:
                break
            if boxes:
                sample = boxes[int(generator.integers(len(boxes)))]
                box_width, box_height = sample[2] - sample[0], sample[3] - sample[1]
            else:
                box_width, box_height = generator.uniform(8, 80, size=2)
            center_x, center_y = generator.uniform(0, width), generator.uniform(0, height)
            box = [center_x - box_width / 2, center_y - box_height / 2,
                   center_x + box_width / 2, center_y + box_height / 2]
            if clear(crop_bounds(box), boxes):
                add(box, 0, 'random_negative', 'background')
                negative_count += 1
        if (image_index + 1) % 20 == 0 or image_index == len(paths) - 1:
            print(f'mined {image_index + 1}/{len(paths)} images: {len(patches)} crops, '
                  f'{dict(Counter(kinds))}', flush=True)
    return np.asarray(patches), np.asarray(labels), np.asarray(kinds), np.asarray(groups), scenes


class CropVerifier:
    def __init__(self, checkpoint):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['patch_size'] != PATCH_SIZE or state['context'] != CONTEXT:
            raise ValueError('Checkpoint crop geometry does not match this verifier')
        self.encoder = Encoder(pretrained=False)
        self.encoder.model.load_state_dict(state['encoder'])
        self.mean = state['mean'].numpy()
        self.scale = state['scale'].numpy()
        self.coef = state['coef'].numpy()
        self.intercept = float(state['intercept'])
        self.threshold = float(state['threshold'])
        self.encoder.features(np.zeros((1, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8))

    def scores(self, features):
        logits = ((features - self.mean) / self.scale) @ self.coef + self.intercept
        return 1 / (1 + np.exp(-np.clip(logits, -60, 60)))

    def rescore(self, image, region, predictions):
        if not predictions:
            return [], []
        patches = [crop(image, view_box(item['bbox'], region, image.shape)) for item in predictions]
        scores = self.scores(self.encoder.features(patches))
        kept = [dict(item, confidence=float(item['confidence'] * score),
                     detector_confidence=float(item['confidence']), verifier_score=float(score))
                for item, score in zip(predictions, scores) if score >= self.threshold]
        return kept, scores.tolist()


class VerifiedDetector:
    """Apply crop rejection before tracking, on the thread warmed at startup."""

    def __init__(self, detector, checkpoint, threshold=None):
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError('Verifier threshold must be finite and between 0 and 1')
        self.detector = detector
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.last_error = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='crop-verifier')
        try:
            self.verifier = self._pool.submit(CropVerifier, self.checkpoint).result()
            self.checkpoint_threshold = self.verifier.threshold
            if threshold is not None:
                self.verifier.threshold = float(threshold)
        except Exception:
            self._pool.shutdown(wait=True)
            raise

    def __call__(self, image, region):
        try:
            predictions = self.detector(image, region)
            kept, _ = self._pool.submit(self.verifier.rescore, image, region, predictions).result()
            self.last_error = None
            return [{'object_id': item['object_id'], 'bbox': item['bbox'],
                     'confidence': item['confidence']} for item in kept]
        except Exception as error:
            self.last_error = type(error).__name__
            raise

    def health(self):
        return {'enabled': True, 'loaded': True, 'checkpoint': str(self.checkpoint),
                'sha256': self.checkpoint_sha256, 'threshold': self.verifier.threshold,
                'checkpoint_threshold': self.checkpoint_threshold,
                'device': self.verifier.encoder.device, 'last_error': self.last_error,
                'mode': 'reject below threshold; multiply detector confidence by verifier score'}


def separation(labels, scores, kinds, groups, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score

    result = {'binary_ap': float(average_precision_score(labels, scores)),
              'roc_auc': float(roc_auc_score(labels, scores)), 'threshold': threshold,
              'counts': dict(Counter(kinds.tolist())), 'accept_rates': {}, 'per_class_retention': {}}
    for kind in sorted(set(kinds)):
        result['accept_rates'][str(kind)] = float(np.mean(scores[kinds == kind] >= threshold))
    for name in sorted(set(groups[labels == 1])):
        result['per_class_retention'][str(name)] = float(np.mean(scores[groups == name] >= threshold))
    return result


def detector_ap(scenes, verifier):
    import local_evaluator

    truth, baseline, filtered = {}, {}, {}
    elapsed = []
    for index, scene in enumerate(scenes):
        image = cv2.imread(scene['image'])
        height, width = image.shape[:2]
        found = scene['predictions']
        started = time.perf_counter()
        rescored, _ = verifier.rescore(image, (0, 0, width, height), found)
        elapsed.append((time.perf_counter() - started) * 1000)
        truth[index], baseline[index], filtered[index] = scene['truth'], found, rescored
    with patch.object(local_evaluator, 'frame_numbers', return_value=list(truth)), \
            patch.object(local_evaluator, 'load_annotations', side_effect=lambda frame, scene: truth[frame]):
        baseline_ap, baseline_classes = local_evaluator.score('synthetic_verifier', baseline)
        filtered_ap, filtered_classes = local_evaluator.score('synthetic_verifier', filtered)
    return {'baseline_ap50': baseline_ap, 'filtered_ap50': filtered_ap,
            'baseline_per_class': baseline_classes, 'filtered_per_class': filtered_classes,
            'verifier_ms_median': float(np.median(elapsed)), 'verifier_ms_p95': float(np.percentile(elapsed, 95))}


def fit(arguments):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from solution import Detector

    if arguments.train_images < 1 or arguments.val_images < 4:
        raise ValueError('Need training images and at least four validation images')
    if arguments.out.exists():
        raise FileExistsError(f'Use a new output directory to preserve the previous experiment: {arguments.out}')
    arguments.out.mkdir(parents=True)
    generator = np.random.default_rng(0)
    train_paths = sorted((DATASET / 'images' / 'train').glob('*.jpg'))
    val_paths = sorted((DATASET / 'images' / 'val').glob('*.jpg'))
    if len(train_paths) < arguments.train_images or len(val_paths) < arguments.val_images:
        raise ValueError('Insufficient images in Helsinki-derived dataset_v5')
    train_paths = [train_paths[index] for index in generator.permutation(len(train_paths))[:arguments.train_images]]
    val_paths = [val_paths[index] for index in generator.permutation(len(val_paths))[:arguments.val_images]]
    calibration_paths = val_paths[:len(val_paths) // 2]
    test_paths = val_paths[len(val_paths) // 2:]
    assert not set(train_paths) & set(val_paths)
    assert not set(calibration_paths) & set(test_paths)
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    detector = Detector(weights=str(arguments.weights.resolve()), confidence=0.05)
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    encoder = Encoder()
    datasets = []
    for name, paths, seed in [('train', train_paths, 0), ('calibration', calibration_paths, 1), ('test', test_paths, 2)]:
        print(f'Preparing {name} from dataset_v5/{paths[0].parent.name}', flush=True)
        patches, labels, kinds, groups, scenes = mine(paths, detector, seed)
        features = encoder.features(patches)
        del patches
        datasets.append((features, labels, kinds, groups, scenes))
        print(f'{name}: {features.shape[0]} frozen feature vectors', flush=True)
    train_features, train_labels, _, _, _ = datasets[0]
    scaler = StandardScaler().fit(train_features)
    classifier = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced', random_state=0)
    classifier.fit(scaler.transform(train_features), train_labels)
    calibration = datasets[1]
    calibration_scores = classifier.predict_proba(scaler.transform(calibration[0]))[:, 1]
    threshold = float(np.quantile(calibration_scores[calibration[1] == 1], 0.05))
    checkpoint = arguments.out / 'verifier.pt'
    torch.save({'encoder': {key: value.cpu() for key, value in encoder.model.state_dict().items()},
                'mean': torch.tensor(scaler.mean_, dtype=torch.float32),
                'scale': torch.tensor(scaler.scale_, dtype=torch.float32),
                'coef': torch.tensor(classifier.coef_[0], dtype=torch.float32),
                'intercept': float(classifier.intercept_[0]), 'threshold': threshold,
                'patch_size': PATCH_SIZE, 'context': CONTEXT}, checkpoint)
    verifier = CropVerifier(checkpoint)
    test_features, test_labels, test_kinds, test_groups, test_scenes = datasets[2]
    test_scores = verifier.scores(test_features)
    reference_scores = classifier.predict_proba(scaler.transform(test_features))[:, 1]
    np.testing.assert_allclose(test_scores, reference_scores, atol=1e-5)
    metrics = separation(test_labels, test_scores, test_kinds, test_groups, threshold)
    metrics['detector'] = detector_ap(test_scenes, verifier)
    metrics['checkpoint'] = str(checkpoint)
    metrics['dataset'] = str(DATASET)
    metrics['split_note'] = 'Disjoint images/seeds, shared Helsinki scenery; NOT evidence of city transfer.'
    metrics['score_note'] = 'Balanced-class logistic scores are not deployment-calibrated probabilities.'
    metrics['detector_weights'] = str(arguments.weights.resolve())
    metrics['detector_sha256'] = hashlib.sha256(arguments.weights.read_bytes()).hexdigest()
    metrics['files'] = {'train': [str(path) for path in train_paths],
                        'calibration': [str(path) for path in calibration_paths],
                        'test': [str(path) for path in test_paths]}
    (arguments.out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in metrics.items() if key != 'files'}, indent=2), flush=True)


def preview(arguments):
    from inspect_recording import draw_inference

    manifest = json.loads(arguments.manifest.read_text(encoding='utf-8'))
    verifier = CropVerifier(arguments.checkpoint)
    arguments.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for item in manifest['frames']:
        image = cv2.imread(item['input'])
        if image is None:
            raise ValueError(f'Cannot read {item["input"]}')
        predictions = item['predictions']
        kept, scores = verifier.rescore(image, item['source_region_xyxy'], predictions)
        meta = dict(item, resolution_level=item['level'])
        before, shown_before = draw_inference(image, meta, predictions, manifest['weights'], 0.30)
        after, shown_after = draw_inference(image, meta, kept, 'v4s1 + crop verifier', 0.30)
        height = max(before.shape[0], after.shape[0])
        panels = [cv2.copyMakeBorder(panel, 0, height - panel.shape[0], 0, 0,
                                     cv2.BORDER_CONSTANT, value=(24, 24, 24)) for panel in (before, after)]
        output = arguments.out / f'frame_{item["frame"]:06d}_comparison.png'
        if not cv2.imwrite(str(output), cv2.vconcat(panels)):
            raise OSError(f'Cannot write {output}')
        summaries.append(dict(frame=item['frame'], raw=len(predictions), accepted=len(kept),
                              shown_before=shown_before, shown_after=shown_after,
                              verifier_scores=scores, predictions=kept, output=str(output)))
        print(f'frame {item["frame"]}: accepted {len(kept)}/{len(predictions)}; '
              f'at display 0.30: {shown_before} -> {shown_after}', flush=True)
    (arguments.out / 'predictions.json').write_text(json.dumps(
        {'checkpoint': str(arguments.checkpoint), 'threshold': verifier.threshold,
         'note': 'Offline inference only. No recording pixels or labels used for fitting or threshold selection.',
         'frames': summaries}, indent=2), encoding='utf-8')


def checks():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[30:50, 60:80] = (0, 0, 255)
    assert crop(image, [60, 30, 80, 50]).shape == (PATCH_SIZE, PATCH_SIZE, 3)
    assert crop(image, [0, 0, 4, 4]).shape == (PATCH_SIZE, PATCH_SIZE, 3)
    assert clear([0, 0, 10, 10], [[11, 0, 20, 10]])
    assert not clear([0, 0, 100, 100], [[40, 40, 41, 41]])
    assert not clear(crop_bounds([10, 10, 20, 20]), [[8, 10, 10, 20]])
    np.testing.assert_allclose(view_box([1000, 600, 1100, 700], [960, 540, 2880, 1620],
                                       (540, 960, 3)), [20, 30, 70, 80])
    import threading

    seen_threads = []

    class StubVerifier:
        def __init__(self, checkpoint):
            seen_threads.append(threading.get_ident())
            self.threshold = 0.179

        def rescore(self, image, region, predictions):
            seen_threads.append(threading.get_ident())
            return [dict(item, confidence=item['confidence'] * 0.8, verifier_score=0.8)
                    for item in predictions if item['confidence'] > 0.5], []

    candidates = [{'object_id': 'hangar', 'bbox': [1, 2, 10, 20], 'confidence': 0.9},
                  {'object_id': 'tank', 'bbox': [20, 30, 40, 50], 'confidence': 0.2}]
    with patch.object(sys.modules[__name__], 'CropVerifier', StubVerifier):
        wrapped = VerifiedDetector(lambda image, region: candidates, Path(__file__))
        try:
            result = wrapped(image, (0, 0, 200, 100))
            assert len(result) == 1 and result[0]['bbox'] == candidates[0]['bbox']
            assert set(result[0]) == {'object_id', 'bbox', 'confidence'}
            assert abs(result[0]['confidence'] - 0.72) < 1e-8
            assert candidates[0]['confidence'] == 0.9
            assert seen_threads[0] == seen_threads[1] != threading.get_ident()
        finally:
            wrapped._pool.shutdown(wait=True)
        wrapped = VerifiedDetector(lambda image, region: candidates, Path(__file__), threshold=0.5)
        try:
            assert wrapped.verifier.threshold == 0.5
            assert wrapped.checkpoint_threshold == 0.179
        finally:
            wrapped._pool.shutdown(wait=True)
    print('PASS: crop shape/borders, conservative target exclusion, source-to-view mapping.')
    print('PASS: verifier rejects/rescores before tracking and uses its warmed worker thread.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('check')
    train = subparsers.add_parser('fit')
    train.add_argument('--weights', type=Path, default=LAB.parent / 'model' / 'v4s1.pt')
    train.add_argument('--train-images', type=int, default=240)
    train.add_argument('--val-images', type=int, default=80)
    train.add_argument('--out', type=Path, default=LAB / 'out' / 'verifier_v1')
    render = subparsers.add_parser('preview')
    render.add_argument('--manifest', type=Path, required=True)
    render.add_argument('--checkpoint', type=Path, required=True)
    render.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == 'check':
        checks()
    elif arguments.command == 'fit':
        fit(arguments)
    else:
        preview(arguments)


if __name__ == '__main__':
    main()