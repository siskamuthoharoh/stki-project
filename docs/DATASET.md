# SkinFind Dataset Documentation



## 1. Dataset Information



- Dataset name: Beauty Product Review

- Authors: Mahendra Dwifebri Purbolaksono, Adiwijaya, and Said Al Faraby

- Publication year: 2022

- Source: Telkom University Dataverse

- Dataset DOI: https://doi.org/10.34820/FK2/NAZYE1

- File DOI: https://doi.org/10.34820/FK2/NAZYE1/MEUDGP

- Local filename: data/raw/produk_kecantikan.csv

- Number of records: 3,960

- Number of columns: 6

- File size: approximately 1.61 MB



## 2. Dataset Description



The dataset contains Indonesian beauty product reviews. Each review has a unique identifier, review text, and labels for four aspects: price, packaging, product, and aroma.



The aspect labels represent sentiment categories:

- -1: Negative

- 0: Neutral

- 1: Positive



## 3. Data Dictionary



| Column | Description |

|---|---|

| review_id | Unique review identifier |

| review_text | Original review text |

| price | Sentiment label for price |

| packaging | Sentiment label for packaging |

| product | Sentiment label for the product |

| aroma | Sentiment label for aroma |



## 4. Initial Validation



The initial validation found:

- 3,960 records and 6 columns.

- No duplicate review IDs.

- No duplicate review texts.

- No missing or empty review texts.

- One missing value in the product label.

- Review length ranges from 29 to 3,411 characters.



The missing product label will remain missing in the raw dataset. It will be excluded from product-aspect sentiment evaluation where a valid label is required.



## 5. Data Processing Policy



The raw dataset will be preserved unchanged. Cleaning and preprocessing will be performed on a separate processed dataset.



Text preprocessing and retrieval results will be documented so that the workflow can be inspected and reproduced.



## 6. Citation



Purbolaksono, M. D., Adiwijaya, and Said Al Faraby. (2022). Beauty Product Review. Telkom University Dataverse. https://doi.org/10.34820/FK2/NAZYE1
