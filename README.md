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
from preprocessing import DataLoader, DataPreProcessor

data_loader = DataLoader()

path = 'data/hackathon_kpis_anonymised-1.csv'

(train, test, val) = data_loader.load(path= path)



# using Bulder pattern to process data. 
df = DataPreProcessor().read_file(read_kpis_path=path).extract_cell_name_data().fix_failure_rates().fetch_data()

print(df.head())

```

### DeepAnT
### isolationForest
### STD

## Contributing

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)