# Sales-Potential-Estimation-Data-Quality-Visualization
=====================================================

This project processes economic indicators from two U.S. sources, enriching them with expert knowledge. 
It prioritizes features, analyzes clusters of U.S. counties, and identifies potential sales regions 
for specific products across the country.


## Projectstructur
---------------


Sales-Potential-Estimation-Data-Quality-Visualization/ ├── notebooks_and_data/ │ ├── data/ │ │ ├── original_data/ │ │ │ ├── pkl/ # Originaldaten im Pickle-Format │ │ │ ├── txt/ # Originaldaten im Textformat │ │ │ └── xlsx/ # Originaldaten im Excel-Format │ │ ├── processed_data/ │ │ │ ├── batch_ids/ # Batch-IDs für die Verarbeitung │ │ │ ├── json/ # Verarbeitete Daten im JSON-Format │ │ │ ├── melted_data/ # Umgeformte (melted) Daten │ │ │ └── pkl/ # Verarbeitete Daten im Pickle-Format │ ├── my_functions/ │ │ ├── functions_data_enrichment.py # Funktionen zur Datenanreicherung │ │ ├── functions_data_linking.py # Funktionen zur Datenverknüpfung │ │ ├── functions_feature_selection.py # Funktionen zur Merkmalsauswahl │ │ └── functions_tsne_analysis.py # Funktionen zur t-SNE-Analyse │ ├── data_linking_ipynb/ # Jupyter Notebook zur Datenverknüpfung │ ├── data_enrichment_ipynb/ # Jupyter Notebook zur Datenanreicherung │ ├── feature_selection_ipynb/ # Jupyter Notebook zur Merkmalsauswahl │ └── t-sne_analysis_ipynb/ # Jupyter Notebook zur t-SNE-Analyse │ └── README.md


## Notebooks (It makes sense to run the notebooks in the mentioned order, as they build on each other)
----------
### 1. `data_linking_ipynb`
External data sources included labor market features for industries (NAICS Code) and occupations (SOC Code) across various U.S. regional dimensions. Key datasets were the 2022 Occupational Employment and Wage Statistics (OEWS) from the Bureau of Labor Statistics and the 2021 County Business Patterns (CBP) from the Census Bureau, focusing on employee counts and establishment numbers across industries and occupations.

The primary challenge was that occupation statistics were only available and comprehensive at the state level. However, the occupation dataset provides the national distribution of occupations by NAICS industry. To address this, we used the national-level distribution of occupations for different industry sectors and the business pattern data to allocate occupation employment numbers from the state level down to individual counties within each state.

Due to the data size, the two datasets need to be manually downloaded and added to the original_data folder. For more details, please refer to the README in the original_data folder.

The occupation_master.xlsx file must be manually adjusted to align with your specific business case and domain field. It is essential to prioritize occupations based on relevance to your industry and the data you are working with. This ensures that the dataset accurately reflects the most important occupations for your analysis, facilitating a more targeted and meaningful application of labor market features.


### 2. `data_enrichment_ipynb`
This notebook leverages ChatGPT's batch API to provide estimates of product consumption for specific products within particular industries and sectors. Tailored prompts have been developed to ensure precise and relevant calculations for each context. 

Additionally, the relevant occupations must be manually prioritized in the occu.xlsx file to highlight the most important occupations for your analysis.

For confidentiality reasons, the prompts have been anonymized and will need to be manually adjusted to fit the specific use case.

### 3. `feature_selection_ipynb/`
This notebook uses a weighted ranking approach to select the most relevant features for industries and occupations, with a focus on expertise and modeling requirements. From a domain perspective, industries and occupations with high employment and significant tool consumption were prioritized. In terms of modeling, the goal was to cover as many regions as possible to optimize clustering. Some industries were manually selected and aggregated due to their strong connection to the manufacturing sector.

Due to anonymization in the enrichment process, the key features were manually selected at the end of the notebook.

### 4. `t-sne_analysis_ipynb/`
This notebook analyzes the U.S. market by applying t-SNE for dimensionality reduction and DBSCAN for clustering across 3,233 counties. The resulting clusters are optimized based on silhouette scores and analyzed to identify key features. The notebook then visualizes the cluster characteristics and regional distributions. A Random Forest model is used to identify the most important features driving cluster assignments. The results highlight significant patterns and insights related to the U.S. market and its regional variations.


