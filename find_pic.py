# 旧的匹配方法，接口已变
import os
import cv2
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from torchvision.models import ResNet50_Weights
from tqdm import tqdm  # For progress bars

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer to get embeddings
model.eval()

# Image transformation pipeline: resize, normalize (handles different resolutions)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Standard size for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_image_paths(folder_path):
    """Get all image paths and filenames from a folder"""
    valid_ext = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")
    img_paths = []
    file_names = []
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder {folder_path} does not exist!")
        return img_paths, file_names
    for file in os.listdir(folder_path):
        if file.lower().endswith(valid_ext):
            full_path = os.path.join(folder_path, file)
            img_paths.append(full_path)
            file_names.append(file)
    return img_paths, file_names


def preprocess_image(image_path):
    """Preprocess image: load, convert to RGB, apply transforms"""
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        return img_tensor
    except Exception as e:
        print(f"❌ Failed to process {image_path}: {str(e)}")
        return None


def extract_features(img_tensor):
    """Extract deep features using ResNet (robust to language, crops, offsets)"""
    if img_tensor is None:
        return None
    with torch.no_grad():
        features = model(img_tensor).squeeze().cpu().numpy()
    return features


def cosine_similarity(featA, featB):
    """Compute cosine similarity between two feature vectors"""
    if featA is None or featB is None:
        return 0.0
    return np.dot(featA, featB) / (np.linalg.norm(featA) * np.linalg.norm(featB))


def mask_text(gray_img):
    """Simple text masking to reduce language impact: blur potential text areas"""
    if gray_img is None:
        return None
    # Detect edges for text (Sobel)
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.uint8(np.clip(np.sqrt(sobelx ** 2 + sobely ** 2), 0, 255))
    _, thresh = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    masked = gray_img.copy()
    text_area = dilated == 255
    blurred = cv2.GaussianBlur(masked, (15, 15), 0)
    masked[text_area] = blurred[text_area]
    return masked


def align_images(imgA, imgB):
    """Basic alignment using feature matching to handle offsets/crops"""
    if imgA is None or imgB is None:
        return imgB
    orb = cv2.ORB_create(nfeatures=1000)
    kpA, desA = orb.detectAndCompute(imgA, None)
    kpB, desB = orb.detectAndCompute(imgB, None)
    if desA is None or desB is None:
        return imgB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desA, desB, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return imgB
    src_pts = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # 关键修复：检查M是否有效且维度正确
    if M is None or M.shape != (3, 3) or not (M.dtype == np.float32 or M.dtype == np.float64):
        return imgB  # 返回原始图像，不进行变换

    aligned = cv2.warpPerspective(imgB, M, (imgA.shape[1], imgA.shape[0]))
    return aligned


def advanced_preprocess(image_path):
    """Advanced preprocess: grayscale, mask text, extract edges for alignment"""
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.resize(img, (512, 512))  # Smaller for alignment
        masked = mask_text(img)
        return masked
    except:
        return None


def calc_similarity_worker(args):
    """Worker for parallel similarity computation"""
    a_feat, b_feat, a_gray, b_gray, b_name, threshold, use_align = args
    sim = cosine_similarity(a_feat, b_feat)
    if sim >= threshold and use_align:
        b_gray_aligned = align_images(a_gray, b_gray)
        aligned_feat = extract_features(
            transform(Image.fromarray(b_gray_aligned).convert("RGB")).unsqueeze(0).to(device))
        sim = cosine_similarity(a_feat, aligned_feat)  # Recompute with aligned
    return b_name, sim


def match_comics(folder_A, folder_B, similarity_threshold=0.85, debug_mode=True,
                 max_workers=None, use_align=True):
    """Main matching function using deep features for high accuracy"""
    a_paths, a_names = get_image_paths(folder_A)
    b_paths, b_names = get_image_paths(folder_B)
    match_result = {name: "" for name in a_names}

    if not a_paths or not b_paths:
        print("⚠️ No valid images in folders!")
        return match_result

    # Pre-extract features for B (efficient)
    print(f"\n🔍 Extracting features for B folder [{len(b_paths)} images]...")
    b_features = []
    b_grays = [] if use_align else None
    for path in tqdm(b_paths):
        tensor = preprocess_image(path)
        feat = extract_features(tensor)
        b_features.append(feat)
        if use_align:
            gray = advanced_preprocess(path)
            b_grays.append(gray)

    # Match each A
    for a_idx, (a_path, a_name) in enumerate(zip(a_paths, a_names), 1):
        if debug_mode:
            print(f"\n📌 Progress [{a_idx}/{len(a_paths)}]: {a_name}")
        a_tensor = preprocess_image(a_path)
        a_feat = extract_features(a_tensor)
        a_gray = advanced_preprocess(a_path) if use_align else None

        best_sim = 0.0
        best_b_name = ""
        valid_matches = []

        # Parallel computation
        with ThreadPoolExecutor(max_workers=4) as executor:
            args_list = [(a_feat, b_feat, a_gray, b_gray, b_name, similarity_threshold / 1.1, use_align)
                         # Slightly lower for candidate selection
                         for b_feat, b_gray, b_name in zip(b_features, b_grays or [None] * len(b_features), b_names)]
            futures = [executor.submit(calc_similarity_worker, args) for args in args_list]
            for future in as_completed(futures):
                b_name, sim = future.result()
                if sim >= similarity_threshold:
                    valid_matches.append((b_name, sim))
                    if sim > best_sim:
                        best_sim = sim
                        best_b_name = b_name

        if debug_mode and valid_matches:
            print(f"  📊 Valid matches (threshold={similarity_threshold}):")
            for bn, s in sorted(valid_matches, key=lambda x: -x[1]):
                print(f"    └─ {bn} | Similarity={s:.4f}")

        if best_b_name:
            match_result[a_name] = best_b_name

    return {'match_result': match_result, 'a_num': len(a_paths), 'b_num': len(b_paths), 'b_names': b_names}


# -------------------------- Configuration --------------------------
if __name__ == "__main__":
    FOLDER_A = r"F:\JHenTai_data\single_Pic\为什么老师会在这里\12_work"  # e.g., Japanese raw
    FOLDER_B = r"F:\JHenTai_data\single_Pic\为什么老师会在这里\12_work\111\汉化"  # e.g., Chinese translated

    SIMILARITY_THRESHOLD = 0.5  # Tune: higher for stricter matches (0.85-0.95 typical for high precision)
    DEBUG_MODE = True
    MAX_WORKERS = None  # Use all CPU cores
    USE_ALIGN = True  # Enable alignment for offsets/crops (slightly slower but better accuracy)

    final_result = match_comics(
        FOLDER_A, FOLDER_B,
        similarity_threshold=SIMILARITY_THRESHOLD,
        debug_mode=DEBUG_MODE,
        max_workers=MAX_WORKERS,
        use_align=USE_ALIGN
    )['match_result']

    print("\n" + "=" * 80)
    print("🎯 Final Matching Results:")
    print(final_result)
    final_result_res = {}
    for k_ in final_result.keys():
        if final_result[k_]:
            final_result_res[k_] = final_result[k_]
    print(final_result_res)
    # Optional: Save to JSON
    # with open("match_results.json", "w", encoding="utf-8") as f:
    #     json.dump(final_result, f, ensure_ascii=False, indent=2)
