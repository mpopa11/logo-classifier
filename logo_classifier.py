import pyarrow.parquet as pq
import pandas as pd
import csv 
import requests
import pandas as pd
import os
import threading
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import cv2
import networkx as nx
import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import DBSCAN

data = pq.read_table('logos.snappy.parquet')
df = data.to_pandas()

df.to_csv('sites.csv', index=False, header=False)

os.makedirs("logos", exist_ok=True)
csv_file = "logos_status.csv"

pd.DataFrame(columns=["Site", "Logo Found", "Downloaded"]).to_csv(csv_file, index=False, mode="w")

lock = threading.Lock()

def get_logo_by_scraping(site):
    site_url = "http://www." + site
    print(f"Scraping {site_url}")

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--blink-settings=imagesEnabled=false")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(15)

    try:
        driver.get(site_url)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        logo = (
            soup.find("img", {"alt": lambda x: x and "logo" in x.lower()}) or
            soup.find("img", {"class": lambda x: x and "logo" in x.lower()}) or
            soup.find("img", src=lambda x: x and "logo" in x.lower())
        )

        if logo:
            logo_url = urljoin(site_url, logo["src"])
            filename = f"logos_safe/{site}.{logo_url.rsplit('.', 1)[-1]}"

            response = requests.get(logo_url, stream=True)
            if response.status_code == 200:
                with open(filename, "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"Downloaded: {filename}")
                return [site, "Found", "Downloaded"]
            else:
                print(f"Failed to download: {site}")
                return [site, "Found", "Failed"]
        else:
            print(f"No logo found for {site} via scraping")
            return [site, "Not Found", "N/A"]
    except Exception as e:
        print(f"Error processing {site} via scraping: {e}")
        return [site, "Error", "N/A"]
    finally:
        driver.quit()

def get_clearbit_logo(site):
    site_url = site.strip().lower()
    clearbit_url = f"https://logo.clearbit.com/{site_url}"

    print(f"Fetching logo for {site_url} from Clearbit...")

    try:
        response = requests.get(clearbit_url, stream=True)
        if response.status_code == 200:
            filename = f"logos_safe/{site_url}.png"
            with open(filename, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded: {filename}")
            return [site, "Found", "Downloaded"]
        else:
            print(f"No logo found for {site} on Clearbit")
            return [site, "Not Found", "N/A"]
    except Exception as e:
        print(f"Error processing {site} on Clearbit: {e}")
        return [site, "Error", "N/A"]

def get_logo(site):
    result = get_logo_by_scraping(site)
    if result[1] == "Not Found" or result[1] == "Error":
        result = get_clearbit_logo(site)

    with lock:
        pd.DataFrame([result]).to_csv(csv_file, mode="a", header=False, index=False)

# Clustering using SSIM graphs

def load_and_preprocess(image_path, target_size=(64, 64)):
    """Loads an image, converts it to grayscale, and resizes it."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping unreadable image: {image_path}")
        return None
    return cv2.resize(img, target_size)

def compute_ssim_pair(img1, img2):
    """Computes SSIM score between two images."""
    return ssim(img1, img2)

def process_image_pairs(image_list, image_data, threshold, max_workers=20):
    """Parallel SSIM computation for image pairs."""
    edges = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, img1 in enumerate(image_list):
            for j, img2 in enumerate(image_list[i+1:], start=i+1):
                futures[(img1, img2)] = executor.submit(compute_ssim_pair, image_data[img1], image_data[img2])

        for (img1, img2), future in futures.items():
            ssim_score = future.result()
            if ssim_score > threshold:
                edges.append((img1, img2))

    return edges

def group_logos_by_ssim(folder, threshold=0.75, output_file="logo_groups_test.txt", max_workers=20):
    """Groups logos based on SSIM similarity and saves results to a file."""
    images = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'))]
    if not images:
        print("No logos found in the folder.")
        return []

    image_data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {img: executor.submit(load_and_preprocess, os.path.join(folder, img)) for img in images}
        for img, future in futures.items():
            processed_img = future.result()
            if processed_img is not None:
                image_data[img] = processed_img

    # If no valid images were loaded, exit early
    if not image_data:
        print("No readable images found.")
        return []

    # Build a similarity graph
    G = nx.Graph()
    image_list = list(image_data.keys())

    # Add all images as nodes 
    G.add_nodes_from(image_list)

    # Compute SSIM 
    edges = process_image_pairs(image_list, image_data, threshold, max_workers)
    G.add_edges_from(edges)

    # Extract connected components as groups
    groups = list(nx.connected_components(G))

    # Save groups to a file
    with open(output_file, "w") as f:
        for i, group in enumerate(groups, 1):
            f.write(f"Group {i}: {list(group)}\n")

    print(f"Logo groups saved to {output_file}")
    return groups

# DBSCAN clustering 

def extract_features(image_data):
    """Converts images to feature vectors by flattening."""
    return np.array([img.flatten() for img in image_data.values()])

def cluster_images_dbscan(features, image_list, eps=5.0, min_samples=2):
    """Clusters images using DBSCAN."""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    return clustering.labels_

def save_clusters(folder, image_list, labels):
    """Saves the clusters as text files with image lists for each groups"""
    output_dir = os.path.join(os.path.dirname(folder), "clusters_dbscan.txt")
    
    with open(output_dir, "w") as f:
        cluster_id = 0
        for label in sorted(set(labels)):
            group_images = [img for img, lbl in zip(image_list, labels) if lbl == label]
            if label == -1:
                for img in group_images:
                    f.write(f"Group {cluster_id}:{img}\n\n")
                    cluster_id += 1
            else:
                f.write(f"Group {cluster_id}:")
                f.write("\n".join(group_images) + "\n\n")
                cluster_id += 1
            print(f"Group {cluster_id}: {group_images}")

def process_images_dbscan(folder, eps=5.0, min_samples=2):
    """Processes images and clusters them using DBSCAN."""
    images = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'))]
    if not images:
        print("No logos found in the folder.")
        return

    image_data = {img: load_and_preprocess(os.path.join(folder, img)) for img in images}
    image_data = {k: v for k, v in image_data.items() if v is not None}
    if not image_data:
        print("No readable images found.")
        return

    features = extract_features(image_data)
    labels = cluster_images_dbscan(features, list(image_data.keys()), eps, min_samples)
    save_clusters(folder, list(image_data.keys()), labels)
    print("Clusters saved using DBSCAN.")

# Scraping
unique_sites = list(dict.fromkeys(pd.read_csv("sites.csv", header=None).transpose().values[0]))

print(len(unique_sites))

with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    list(executor.map(get_logo, unique_sites))
print("Scraping complete. Results saved in logos_status.csv")

# Grouping by SSIM graph
folder = "logos"
logo_groups = group_logos_by_ssim(folder, threshold=0.7, output_file = "logo_groups_ssim.txt" , max_workers=200)

# Print results
if logo_groups:
    for i, group in enumerate(logo_groups, 1):
        print(f"Group {i}: {list(group)}")
else:
    print("No similar logos found.")

# Clustering using DBSCAN

folder = "logos"
process_images_dbscan(folder, eps=500.0, min_samples=2)
