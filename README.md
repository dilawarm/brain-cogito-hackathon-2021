#  Telenor - Unsupervised  Anomaly Detection for Telenor Network Data
## Hackathon 2021 - Group 5 


## Installation
In this project we used requirements.txt file. To install the packages please write: 

```bash
cd brain-cogito-hackathon-2021/
pip install -r requirements.txt
```


## Usage
To run this project you'll need to have access to 

```python
'hackathon_kpis_anonymised.csv'
'relative_distance.csv'
```
These files can be found at  [Telenor's](https://drive.google.com/drive/folders/1XRphkrv0Lod4awZFCtcQQJx_qagIxsq2) google drive. 

### preprocessing
All actions for this step can be found in the preprocessing.py file located at root folder. 

```python
from preprocessing import DataPreProcessor
from hierarchical_clustering import HierarchicalClustering
import pandas as pd

path_distance = 'data/relative_distance.csv'
path_harc = 'data/cell_clusters.csv'

# preprocessing the data for hierarchical clustering
hiarc_df = HierarchicalClustering.from_path(path = path_distance).cluster_data().extract_cell_name_from_clusters()

hiarc_df.to_csv(path_harc)

path = 'data/hackathon_kpis_anonymised-1.csv'

# preprocessing by extracting each feature from the cell name. 
# fixing the failure rates, for example 0 calls divided by 0 becomes NaN, this is now changed to 0.
# Process path is using hierarchical clustering data from csv. 

df = DataPreProcessor().read_file(read_kpis_path=path).extract_cell_name_data().fix_failure_rates().process_from_path(path_harc).fetch_data()

print(df.head())

```
The example above shows how to use every function in our arsenal to preprocess the data given for this assignment. 
### DeepAnT
### isolationForest
### STD

## Contributing

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)