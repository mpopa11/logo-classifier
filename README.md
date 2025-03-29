# Website Logo Clustering Approach

## **1. Retrieving Websites**

The first step in this task was to retrieve all the unique websites from the provided dataset. With this list, I proceeded to extract logos from each website.

## **2. Extracting Logos**

### **Initial Approach: Web Scraping**
My first instinct was to scrape the websites using Selenium for headless browsing and BeautifulSoup to parse the HTML. I specifically looked for images that contained the keyword **"logo"** in their attributes, such as:
- The `alt` attribute
- The `class` name
- The file name (`src`)

However, I encountered several issues:
- Some websites were **inaccessible** due to restrictions or timeouts.
- Many websites **did not have their logos directly in the HTML**, instead storing them in **CSS files** or external stylesheets.

This initial approach only yielded about **2200** logos out of **3400** unique websites, which was insufficient.

### **Enhancing with Clearbit API**
To increase the number of retrieved logos, I incorporated the **Clearbit API**, which provides logo images for many known websites. This significantly improved my results, allowing me to extract **almost 3300** logos from the dataset.

---

## **3. Grouping Logos**

With all the logos downloaded, I moved on to the grouping phase. My first instinct was to use a **clustering algorithm**. After preprocessing the images (grayscale conversion and resizing), I tested:

### **K-Means Clustering**
- K-Means was not effective, as logos vary significantly in shape, color, and style, making the clustering inaccurate.

### **DBSCAN (Density-Based Clustering)**
- DBSCAN performed better than K-Means, as it does not assume a fixed number of clusters.
- However, it still did not provide **optimal** grouping, and I believed I could improve the approach further.

---

## **4. Exploring Alternative Solutions**

### **Perceptual Hashing (pHash)**
I first experimented with **perceptual hashing (pHash)**, which computes a hash based on the visual content of an image. While promising, it had a major drawback:
- If two logos were identical but had different **backgrounds**, they would not be classified together.
- Attempts to **remove backgrounds** increased computational complexity and did not yield reliable results.
- The added **overhead** made this approach impractical for large datasets.

### **Structural Similarity Index Measure (SSIM)**
Given the challenges with pHash, I decided to use **SSIM (Structural Similarity Index Measure)**, which compares images based on:
- **Luminance**
- **Contrast**
- **Structural components**

SSIM provides a similarity score between **0 and 1**:
- `1` → Identical images
- `0` → Completely different images
#### **SSIM Formula**

For two images **X** and **Y**, SSIM is calculated as:

$$
SSIM(X,Y) = \frac{(2\mu_X\mu_Y + C_1)(2\sigma_{XY} + C_2)}{(\mu_X^2 + \mu_Y^2 + C_1)(\sigma_X^2 + \sigma_Y^2 + C_2)}
$$

--- 

## **5. Graph-Based Clustering with SSIM**

### **Building the Similarity Graph**
- I created a **graph** where each **node** represents a logo.
- If the **SSIM score** between two logos exceeded a threshold (**0.7** proved optimal through experimentation), an **edge** was added between them.

### **Extracting Groups**
- **Connected components** in the graph represent clusters of similar logos.
- A **fully connected component** (where all logos are connected by similarity) forms a **logo group**.

#### **Example Graph Structure**
```
logo1 ---- logo2 ---- logo3
logo4 ---- logo5
logo6 (no connections)
```

This results in three groups:
1. `{logo1, logo2, logo3}`
2. `{logo4, logo5}`
3. `{logo6}` (a unique logo)

---

## **6. Results and Conclusion**

- The **SSIM-based approach** outperformed both **pHash** and classical clustering algorithms.
- It successfully grouped **logos with different backgrounds** without requiring background removal.
- **Connected components** in the SSIM graph naturally formed **accurate logo clusters**.

This method proved to be the most **effective and scalable** solution for the given dataset, providing **better logo similarity grouping than conventional clustering techniques**.
