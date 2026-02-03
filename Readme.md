## AIRCRAFT ACCIDENT ANALYSIS

#3 BUSINESS UNDERSTANDING

Manhattan Limited is planning to  venture into the aviation industry by buying and operating commercial and private aircraft.The company has limited knowledge and clear understanding of the safety and risk profiles of different aircraft.

Aircraft accidents can lead to major financial losses, legal liabilities, damaged reputation and safety risk for passengers and staff.Because of this , it is essential for the company to make decisions based on facts from prior data from the aviation industry to be able to choose which aircraft models to invest in that minimize risks.

This analsis aims to review historical aviation accident data to identify types of aircraft with highest operation risk. These findings will help management pick safer and more reliable aircraft for the company's new aviation division.

## PROJECT OBJECTIVES

1.To analyse aircraft accident trends over time.

2.To identify aircraft types with the highest accidents rates.  

3.To evaluate which type of damage level has the most fatilities.

4.To analyze which operators have the highest accidents.

5.To generate insights and recommendations that management can use to guide aircraft purchasing decisons



## DATA UNDERSTANDING


```python
#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
#loading dataset
df = pd.read_csv('flight.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>acc.date</th>
      <th>type</th>
      <th>reg</th>
      <th>operator</th>
      <th>fat</th>
      <th>location</th>
      <th>dmg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3 Jan 2022</td>
      <td>British Aerospace 4121 Jetstream 41</td>
      <td>ZS-NRJ</td>
      <td>SA Airlink</td>
      <td>0</td>
      <td>near Venetia Mine Airport</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4 Jan 2022</td>
      <td>British Aerospace 3101 Jetstream 31</td>
      <td>HR-AYY</td>
      <td>LANHSA - Línea Aérea Nacional de Honduras S.A</td>
      <td>0</td>
      <td>Roatán-Juan Manuel Gálvez International Airpor...</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5 Jan 2022</td>
      <td>Boeing 737-4H6</td>
      <td>EP-CAP</td>
      <td>Caspian Airlines</td>
      <td>0</td>
      <td>Isfahan-Shahid Beheshti Airport (IFN)</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8 Jan 2022</td>
      <td>Tupolev Tu-204-100C</td>
      <td>RA-64032</td>
      <td>Cainiao, opb Aviastar-TU</td>
      <td>0</td>
      <td>Hangzhou Xiaoshan International Airport (HGH)</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>12 Jan 2022</td>
      <td>Beechcraft 200 Super King Air</td>
      <td>NaN</td>
      <td>private</td>
      <td>0</td>
      <td>Machakilha, Toledo District, Grahem Creek area</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2495</th>
      <td>1245</td>
      <td>20 Dec 2018</td>
      <td>Cessna 560 Citation V</td>
      <td>N188CW</td>
      <td>Chen Aircrafts LLC</td>
      <td>4</td>
      <td>2 km NE of Atlanta-Fulton County Airport, GA (...</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>2496</th>
      <td>1246</td>
      <td>22 Dec 2018</td>
      <td>PZL-Mielec M28 Skytruck</td>
      <td>GNB-96107</td>
      <td>Guardia Nacional Bolivariana de Venezuela - GNBV</td>
      <td>0</td>
      <td>Kamarata Airport (KTV)</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>2497</th>
      <td>1247</td>
      <td>24 Dec 2018</td>
      <td>Antonov An-26B</td>
      <td>9T-TAB</td>
      <td>Air Force of the Democratic Republic of the Congo</td>
      <td>0</td>
      <td>Beni Airport (BNC)</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>2498</th>
      <td>1248</td>
      <td>31 Dec 2018</td>
      <td>Boeing 757-2B7 (WL)</td>
      <td>N938UW</td>
      <td>American Airlines</td>
      <td>0</td>
      <td>Charlotte-Douglas International Airport, NC (C...</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>2499</th>
      <td>1249</td>
      <td>unk. date 2018</td>
      <td>Rockwell Sabreliner 80</td>
      <td>N337KL</td>
      <td>private</td>
      <td>0</td>
      <td>Eugene Airport, OR (EUG)</td>
      <td>sub</td>
    </tr>
  </tbody>
</table>
<p>2500 rows × 8 columns</p>
</div>




```python
# looking at datatypes and missing values
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2500 entries, 0 to 2499
    Data columns (total 8 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   Unnamed: 0  2500 non-null   int64 
     1   acc.date    2500 non-null   object
     2   type        2500 non-null   object
     3   reg         2408 non-null   object
     4   operator    2486 non-null   object
     5   fat         2488 non-null   object
     6   location    2500 non-null   object
     7   dmg         2500 non-null   object
    dtypes: int64(1), object(7)
    memory usage: 156.4+ KB
    


```python
# checking missing value
df.isna().sum()
```




    Unnamed: 0     0
    acc.date       0
    type           0
    reg           92
    operator      14
    fat           12
    location       0
    dmg            0
    dtype: int64



df.columns


```python
#Checking for rows and columns
df.shape
```




    (2500, 8)




```python
# checking for duplicates
df.duplicated().value_counts()

```




    False    1250
    True     1250
    Name: count, dtype: int64



## DATA PREPARATION


```python
#Sorting duplicates
df[df.duplicated(keep=False)].sort_values(by='type')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>acc.date</th>
      <th>type</th>
      <th>reg</th>
      <th>operator</th>
      <th>fat</th>
      <th>location</th>
      <th>dmg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3 Jan 2022</td>
      <td>British Aerospace 4121 Jetstream 41</td>
      <td>ZS-NRJ</td>
      <td>SA Airlink</td>
      <td>0</td>
      <td>near Venetia Mine Airport</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4 Jan 2022</td>
      <td>British Aerospace 3101 Jetstream 31</td>
      <td>HR-AYY</td>
      <td>LANHSA - Línea Aérea Nacional de Honduras S.A</td>
      <td>0</td>
      <td>Roatán-Juan Manuel Gálvez International Airpor...</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5 Jan 2022</td>
      <td>Boeing 737-4H6</td>
      <td>EP-CAP</td>
      <td>Caspian Airlines</td>
      <td>0</td>
      <td>Isfahan-Shahid Beheshti Airport (IFN)</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8 Jan 2022</td>
      <td>Tupolev Tu-204-100C</td>
      <td>RA-64032</td>
      <td>Cainiao, opb Aviastar-TU</td>
      <td>0</td>
      <td>Hangzhou Xiaoshan International Airport (HGH)</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>12 Jan 2022</td>
      <td>Beechcraft 200 Super King Air</td>
      <td>NaN</td>
      <td>private</td>
      <td>0</td>
      <td>Machakilha, Toledo District, Grahem Creek area</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2495</th>
      <td>1245</td>
      <td>20 Dec 2018</td>
      <td>Cessna 560 Citation V</td>
      <td>N188CW</td>
      <td>Chen Aircrafts LLC</td>
      <td>4</td>
      <td>2 km NE of Atlanta-Fulton County Airport, GA (...</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>2496</th>
      <td>1246</td>
      <td>22 Dec 2018</td>
      <td>PZL-Mielec M28 Skytruck</td>
      <td>GNB-96107</td>
      <td>Guardia Nacional Bolivariana de Venezuela - GNBV</td>
      <td>0</td>
      <td>Kamarata Airport (KTV)</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>2497</th>
      <td>1247</td>
      <td>24 Dec 2018</td>
      <td>Antonov An-26B</td>
      <td>9T-TAB</td>
      <td>Air Force of the Democratic Republic of the Congo</td>
      <td>0</td>
      <td>Beni Airport (BNC)</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>2498</th>
      <td>1248</td>
      <td>31 Dec 2018</td>
      <td>Boeing 757-2B7 (WL)</td>
      <td>N938UW</td>
      <td>American Airlines</td>
      <td>0</td>
      <td>Charlotte-Douglas International Airport, NC (C...</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>2499</th>
      <td>1249</td>
      <td>unk. date 2018</td>
      <td>Rockwell Sabreliner 80</td>
      <td>N337KL</td>
      <td>private</td>
      <td>0</td>
      <td>Eugene Airport, OR (EUG)</td>
      <td>sub</td>
    </tr>
  </tbody>
</table>
<p>2500 rows × 8 columns</p>
</div>




```python
#Dropping duplicates
df = df.drop_duplicates()
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>acc.date</th>
      <th>type</th>
      <th>reg</th>
      <th>operator</th>
      <th>fat</th>
      <th>location</th>
      <th>dmg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3 Jan 2022</td>
      <td>British Aerospace 4121 Jetstream 41</td>
      <td>ZS-NRJ</td>
      <td>SA Airlink</td>
      <td>0</td>
      <td>near Venetia Mine Airport</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4 Jan 2022</td>
      <td>British Aerospace 3101 Jetstream 31</td>
      <td>HR-AYY</td>
      <td>LANHSA - Línea Aérea Nacional de Honduras S.A</td>
      <td>0</td>
      <td>Roatán-Juan Manuel Gálvez International Airpor...</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5 Jan 2022</td>
      <td>Boeing 737-4H6</td>
      <td>EP-CAP</td>
      <td>Caspian Airlines</td>
      <td>0</td>
      <td>Isfahan-Shahid Beheshti Airport (IFN)</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8 Jan 2022</td>
      <td>Tupolev Tu-204-100C</td>
      <td>RA-64032</td>
      <td>Cainiao, opb Aviastar-TU</td>
      <td>0</td>
      <td>Hangzhou Xiaoshan International Airport (HGH)</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>12 Jan 2022</td>
      <td>Beechcraft 200 Super King Air</td>
      <td>NaN</td>
      <td>private</td>
      <td>0</td>
      <td>Machakilha, Toledo District, Grahem Creek area</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1245</th>
      <td>1245</td>
      <td>20 Dec 2018</td>
      <td>Cessna 560 Citation V</td>
      <td>N188CW</td>
      <td>Chen Aircrafts LLC</td>
      <td>4</td>
      <td>2 km NE of Atlanta-Fulton County Airport, GA (...</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>1246</th>
      <td>1246</td>
      <td>22 Dec 2018</td>
      <td>PZL-Mielec M28 Skytruck</td>
      <td>GNB-96107</td>
      <td>Guardia Nacional Bolivariana de Venezuela - GNBV</td>
      <td>0</td>
      <td>Kamarata Airport (KTV)</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1247</th>
      <td>1247</td>
      <td>24 Dec 2018</td>
      <td>Antonov An-26B</td>
      <td>9T-TAB</td>
      <td>Air Force of the Democratic Republic of the Congo</td>
      <td>0</td>
      <td>Beni Airport (BNC)</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>1248</th>
      <td>1248</td>
      <td>31 Dec 2018</td>
      <td>Boeing 757-2B7 (WL)</td>
      <td>N938UW</td>
      <td>American Airlines</td>
      <td>0</td>
      <td>Charlotte-Douglas International Airport, NC (C...</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1249</th>
      <td>1249</td>
      <td>unk. date 2018</td>
      <td>Rockwell Sabreliner 80</td>
      <td>N337KL</td>
      <td>private</td>
      <td>0</td>
      <td>Eugene Airport, OR (EUG)</td>
      <td>sub</td>
    </tr>
  </tbody>
</table>
<p>1250 rows × 8 columns</p>
</div>




```python
#Dropping duplicates
df = df.drop_duplicates()
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>acc.date</th>
      <th>type</th>
      <th>reg</th>
      <th>operator</th>
      <th>fat</th>
      <th>location</th>
      <th>dmg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3 Jan 2022</td>
      <td>British Aerospace 4121 Jetstream 41</td>
      <td>ZS-NRJ</td>
      <td>SA Airlink</td>
      <td>0</td>
      <td>near Venetia Mine Airport</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4 Jan 2022</td>
      <td>British Aerospace 3101 Jetstream 31</td>
      <td>HR-AYY</td>
      <td>LANHSA - Línea Aérea Nacional de Honduras S.A</td>
      <td>0</td>
      <td>Roatán-Juan Manuel Gálvez International Airpor...</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5 Jan 2022</td>
      <td>Boeing 737-4H6</td>
      <td>EP-CAP</td>
      <td>Caspian Airlines</td>
      <td>0</td>
      <td>Isfahan-Shahid Beheshti Airport (IFN)</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8 Jan 2022</td>
      <td>Tupolev Tu-204-100C</td>
      <td>RA-64032</td>
      <td>Cainiao, opb Aviastar-TU</td>
      <td>0</td>
      <td>Hangzhou Xiaoshan International Airport (HGH)</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>12 Jan 2022</td>
      <td>Beechcraft 200 Super King Air</td>
      <td>NaN</td>
      <td>private</td>
      <td>0</td>
      <td>Machakilha, Toledo District, Grahem Creek area</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1245</th>
      <td>1245</td>
      <td>20 Dec 2018</td>
      <td>Cessna 560 Citation V</td>
      <td>N188CW</td>
      <td>Chen Aircrafts LLC</td>
      <td>4</td>
      <td>2 km NE of Atlanta-Fulton County Airport, GA (...</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>1246</th>
      <td>1246</td>
      <td>22 Dec 2018</td>
      <td>PZL-Mielec M28 Skytruck</td>
      <td>GNB-96107</td>
      <td>Guardia Nacional Bolivariana de Venezuela - GNBV</td>
      <td>0</td>
      <td>Kamarata Airport (KTV)</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1247</th>
      <td>1247</td>
      <td>24 Dec 2018</td>
      <td>Antonov An-26B</td>
      <td>9T-TAB</td>
      <td>Air Force of the Democratic Republic of the Congo</td>
      <td>0</td>
      <td>Beni Airport (BNC)</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>1248</th>
      <td>1248</td>
      <td>31 Dec 2018</td>
      <td>Boeing 757-2B7 (WL)</td>
      <td>N938UW</td>
      <td>American Airlines</td>
      <td>0</td>
      <td>Charlotte-Douglas International Airport, NC (C...</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1249</th>
      <td>1249</td>
      <td>unk. date 2018</td>
      <td>Rockwell Sabreliner 80</td>
      <td>N337KL</td>
      <td>private</td>
      <td>0</td>
      <td>Eugene Airport, OR (EUG)</td>
      <td>sub</td>
    </tr>
  </tbody>
</table>
<p>1250 rows × 8 columns</p>
</div>




```python
#Remove unnecessary index column 'Unnamed: 0'
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acc.date</th>
      <th>type</th>
      <th>reg</th>
      <th>operator</th>
      <th>fat</th>
      <th>location</th>
      <th>dmg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3 Jan 2022</td>
      <td>British Aerospace 4121 Jetstream 41</td>
      <td>ZS-NRJ</td>
      <td>SA Airlink</td>
      <td>0</td>
      <td>near Venetia Mine Airport</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4 Jan 2022</td>
      <td>British Aerospace 3101 Jetstream 31</td>
      <td>HR-AYY</td>
      <td>LANHSA - Línea Aérea Nacional de Honduras S.A</td>
      <td>0</td>
      <td>Roatán-Juan Manuel Gálvez International Airpor...</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5 Jan 2022</td>
      <td>Boeing 737-4H6</td>
      <td>EP-CAP</td>
      <td>Caspian Airlines</td>
      <td>0</td>
      <td>Isfahan-Shahid Beheshti Airport (IFN)</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8 Jan 2022</td>
      <td>Tupolev Tu-204-100C</td>
      <td>RA-64032</td>
      <td>Cainiao, opb Aviastar-TU</td>
      <td>0</td>
      <td>Hangzhou Xiaoshan International Airport (HGH)</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12 Jan 2022</td>
      <td>Beechcraft 200 Super King Air</td>
      <td>NaN</td>
      <td>private</td>
      <td>0</td>
      <td>Machakilha, Toledo District, Grahem Creek area</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1245</th>
      <td>20 Dec 2018</td>
      <td>Cessna 560 Citation V</td>
      <td>N188CW</td>
      <td>Chen Aircrafts LLC</td>
      <td>4</td>
      <td>2 km NE of Atlanta-Fulton County Airport, GA (...</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>1246</th>
      <td>22 Dec 2018</td>
      <td>PZL-Mielec M28 Skytruck</td>
      <td>GNB-96107</td>
      <td>Guardia Nacional Bolivariana de Venezuela - GNBV</td>
      <td>0</td>
      <td>Kamarata Airport (KTV)</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1247</th>
      <td>24 Dec 2018</td>
      <td>Antonov An-26B</td>
      <td>9T-TAB</td>
      <td>Air Force of the Democratic Republic of the Congo</td>
      <td>0</td>
      <td>Beni Airport (BNC)</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>1248</th>
      <td>31 Dec 2018</td>
      <td>Boeing 757-2B7 (WL)</td>
      <td>N938UW</td>
      <td>American Airlines</td>
      <td>0</td>
      <td>Charlotte-Douglas International Airport, NC (C...</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1249</th>
      <td>unk. date 2018</td>
      <td>Rockwell Sabreliner 80</td>
      <td>N337KL</td>
      <td>private</td>
      <td>0</td>
      <td>Eugene Airport, OR (EUG)</td>
      <td>sub</td>
    </tr>
  </tbody>
</table>
<p>1250 rows × 7 columns</p>
</div>




```python
# Selecting relevant columns

rel_columns = ['acc.date','type','operator','fat','dmg']
df = df[rel_columns]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acc.date</th>
      <th>type</th>
      <th>operator</th>
      <th>fat</th>
      <th>dmg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3 Jan 2022</td>
      <td>British Aerospace 4121 Jetstream 41</td>
      <td>SA Airlink</td>
      <td>0</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4 Jan 2022</td>
      <td>British Aerospace 3101 Jetstream 31</td>
      <td>LANHSA - Línea Aérea Nacional de Honduras S.A</td>
      <td>0</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5 Jan 2022</td>
      <td>Boeing 737-4H6</td>
      <td>Caspian Airlines</td>
      <td>0</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8 Jan 2022</td>
      <td>Tupolev Tu-204-100C</td>
      <td>Cainiao, opb Aviastar-TU</td>
      <td>0</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12 Jan 2022</td>
      <td>Beechcraft 200 Super King Air</td>
      <td>private</td>
      <td>0</td>
      <td>w/o</td>
    </tr>
  </tbody>
</table>
</div>




```python
#naming the columns properly
df.columns = ['acc.date', 'aircraft_type', 'operator', 'fatalities', 'damage']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acc.date</th>
      <th>aircraft_type</th>
      <th>operator</th>
      <th>fatalities</th>
      <th>damage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3 Jan 2022</td>
      <td>British Aerospace 4121 Jetstream 41</td>
      <td>SA Airlink</td>
      <td>0</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4 Jan 2022</td>
      <td>British Aerospace 3101 Jetstream 31</td>
      <td>LANHSA - Línea Aérea Nacional de Honduras S.A</td>
      <td>0</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5 Jan 2022</td>
      <td>Boeing 737-4H6</td>
      <td>Caspian Airlines</td>
      <td>0</td>
      <td>sub</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8 Jan 2022</td>
      <td>Tupolev Tu-204-100C</td>
      <td>Cainiao, opb Aviastar-TU</td>
      <td>0</td>
      <td>w/o</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12 Jan 2022</td>
      <td>Beechcraft 200 Super King Air</td>
      <td>private</td>
      <td>0</td>
      <td>w/o</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove rows where 'acc.date' is empty after cleaning
df = df.copy()
#Ensuring the date is in a string
df['acc.date'] = df['acc.date'].astype(str)
#Removing 'xx' and ' ' and replacing with False
df['acc.date'] = df['acc.date'].str.replace('xx', '', regex=False)
#Removing unk.date, and replacing with false
df['acc.date'] = df['acc.date'].str.replace('unk. date', '', regex=False)
# Removing spaces in our string
df['acc.date'] = df['acc.date'].str.strip()
# Convert to datetime
df['acc.date'] = pd.to_datetime(df['acc.date'], errors='coerce')
#Converting acc.date from object to datetime
# Drop rows where conversion failed (NaT)
df = df.dropna(subset=['acc.date'])
df['acc.date']

```




    0      2022-01-03
    1      2022-01-04
    2      2022-01-05
    3      2022-01-08
    4      2022-01-12
              ...    
    1244   2018-12-20
    1245   2018-12-20
    1246   2018-12-22
    1247   2018-12-24
    1248   2018-12-31
    Name: acc.date, Length: 1247, dtype: datetime64[ns]




```python
# Extract year from 'acc.date' and create a new column
df['year'] = df['acc.date'].dt.year
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acc.date</th>
      <th>aircraft_type</th>
      <th>operator</th>
      <th>fatalities</th>
      <th>damage</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-01-03</td>
      <td>British Aerospace 4121 Jetstream 41</td>
      <td>SA Airlink</td>
      <td>0</td>
      <td>sub</td>
      <td>2022</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-01-04</td>
      <td>British Aerospace 3101 Jetstream 31</td>
      <td>LANHSA - Línea Aérea Nacional de Honduras S.A</td>
      <td>0</td>
      <td>sub</td>
      <td>2022</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-01-05</td>
      <td>Boeing 737-4H6</td>
      <td>Caspian Airlines</td>
      <td>0</td>
      <td>sub</td>
      <td>2022</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-01-08</td>
      <td>Tupolev Tu-204-100C</td>
      <td>Cainiao, opb Aviastar-TU</td>
      <td>0</td>
      <td>w/o</td>
      <td>2022</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-01-12</td>
      <td>Beechcraft 200 Super King Air</td>
      <td>private</td>
      <td>0</td>
      <td>w/o</td>
      <td>2022</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1244</th>
      <td>2018-12-20</td>
      <td>Antonov An-26B</td>
      <td>Gomair</td>
      <td>7</td>
      <td>w/o</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1245</th>
      <td>2018-12-20</td>
      <td>Cessna 560 Citation V</td>
      <td>Chen Aircrafts LLC</td>
      <td>4</td>
      <td>w/o</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1246</th>
      <td>2018-12-22</td>
      <td>PZL-Mielec M28 Skytruck</td>
      <td>Guardia Nacional Bolivariana de Venezuela - GNBV</td>
      <td>0</td>
      <td>sub</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1247</th>
      <td>2018-12-24</td>
      <td>Antonov An-26B</td>
      <td>Air Force of the Democratic Republic of the Congo</td>
      <td>0</td>
      <td>w/o</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1248</th>
      <td>2018-12-31</td>
      <td>Boeing 757-2B7 (WL)</td>
      <td>American Airlines</td>
      <td>0</td>
      <td>sub</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
<p>1247 rows × 6 columns</p>
</div>




```python
#Remove spaces
df['aircraft_type'] = df['aircraft_type'].str.strip() 
# Standardize Capitalization
df['aircraft_type'] = df['aircraft_type'].str.title() 
df['aircraft_type']
```




    0       British Aerospace 4121 Jetstream 41
    1       British Aerospace 3101 Jetstream 31
    2                            Boeing 737-4H6
    3                       Tupolev Tu-204-100C
    4             Beechcraft 200 Super King Air
                           ...                 
    1244                         Antonov An-26B
    1245                  Cessna 560 Citation V
    1246                Pzl-Mielec M28 Skytruck
    1247                         Antonov An-26B
    1248                    Boeing 757-2B7 (Wl)
    Name: aircraft_type, Length: 1247, dtype: object




```python
#Cleaning Operator column
#Remove spaces
df['operator'] = df['operator'].str.strip() 
# Standardize Capitalization
df['operator'] = df['operator'].str.title() 
df['operator'] = df['operator'].fillna('unknown')
df['operator']
```




    0                                              Sa Airlink
    1           Lanhsa - Línea Aérea Nacional De Honduras S.A
    2                                        Caspian Airlines
    3                                Cainiao, Opb Aviastar-Tu
    4                                                 Private
                                  ...                        
    1244                                               Gomair
    1245                                   Chen Aircrafts Llc
    1246     Guardia Nacional Bolivariana De Venezuela - Gnbv
    1247    Air Force Of The Democratic Republic Of The Congo
    1248                                    American Airlines
    Name: operator, Length: 1247, dtype: object




```python
# Cleaning Fatalities column
#Converting from a string to numeric, filling missing values and making it an integer
df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0).astype(int)
df['fatalities']
```




    0       0
    1       0
    2       0
    3       0
    4       0
           ..
    1244    7
    1245    4
    1246    0
    1247    0
    1248    0
    Name: fatalities, Length: 1247, dtype: int32




```python
# Strip whitespace and lowercase
df['damage'] = df['damage'].str.strip().str.lower()

# Replace shorthand codes with full,standardized labels
df['damage'] = df['damage'].replace({'sub': 'Substantial','w/o': 'Written Off','non': 'None','min': 'Minor','mis':'Missing','unk':'Unknown'})

#Fill any missing values
df['damage'] = df['damage'].fillna('Unknown')
df['damage']
```




    0       Substantial
    1       Substantial
    2       Substantial
    3       Written Off
    4       Written Off
               ...     
    1244    Written Off
    1245    Written Off
    1246    Substantial
    1247    Written Off
    1248    Substantial
    Name: damage, Length: 1247, dtype: object




```python
df.to_csv("aircraft_accidents_cleaned.csv", index=False)

```

DATA ANALYSIS


```python
#Grouping  the data by year
accidents_per_year = df.groupby(df['acc.date'].dt.year).size()
accidents_per_year.head()

```




    acc.date
    2018    284
    2019    295
    2020    234
    2021    216
    2022    218
    dtype: int64




```python
# Plotting how many accidents were there each year to see the trend
plt.figure(figsize=(10,6))
# Line with markers
plt.plot(accidents_per_year.index, accidents_per_year.values, marker='*', color='blue', linewidth=1)
# Labels and title
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.title('Aircraft Accidents per Year')
# Make x-axis show all years 
plt.xticks(accidents_per_year.index, rotation=90)
# grid
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

```


    
![png](output_26_0.png)
    



```python
accidents_per_type = df['aircraft_type'].value_counts()
accidents_per_type
```




    aircraft_type
    Cessna 208B Grand Caravan                   57
    Beechcraft 200 Super King Air               29
    Antonov An-2R                               28
    De Havilland Canada Dhc-6 Twin Otter 300    17
    Cessna 208 Caravan I                        15
                                                ..
    Boeing 767-375Er                             1
    Boeing 747-412 (Bcf)                         1
    Boeing 747-412F (Scd)                        1
    Boeing 737-76N (Wl)                          1
    Boeing 757-2B7 (Wl)                          1
    Name: count, Length: 522, dtype: int64




```python
# Identify aircraft types with the highest accidents rates .
#Count how many accidents each aircraft type has
accidents_per_type = df['aircraft_type'].value_counts()
#Select the top 10 aircraft types with the highest accident counts
top_10_aircraft = accidents_per_type.head(10)
# Plot a horizontal bar chart 
plt.figure(figsize=(12,6))
top_10_aircraft.sort_values().plot(kind='bar', color='black')
plt.title('Top 10 Aircraft Types with Highest Accident Counts')
plt.xlabel('Aircraft Type')
plt.ylabel('Number of Accidents')
plt.show()
```


    
![png](output_28_0.png)
    



```python
# which operators had the highest accidents

accidents_per_operator = df['operator'].value_counts()
top_10_operators = accidents_per_operator.head(10)
plt.figure(figsize=(12,6))
top_10_operators.sort_values().plot(kind='bar', color='skyblue') 
plt.title('Top 10 Operators by Number of Accidents', fontsize=16)
plt.xlabel('Operator', fontsize=12)
plt.ylabel('Number of Accidents', fontsize=12) 
plt.show()


```


    
![png](output_29_0.png)
    



```python
#evaluate the fatalities with the type of damage level
fatalities= df['fatalities'].value_counts()
#Total fatalities per aircraft type
fatalities_by_damage = df.groupby('damage')['fatalities'].sum().sort_values(ascending=False)
# Total accidents by damage ype
damage_summary = df['damage'].value_counts()

# Visualize
plt.figure(figsize=(10,5))
plt.bar(x=fatalities_by_damage.index, height=fatalities_by_damage.values, color='red')
plt.title("Total Fatalities by Damage Level")
plt.xlabel('Damage Type')
plt.ylabel("Total Fatalities")
plt.show()

```


    
![png](output_30_0.png)
    


FINDINGS 

1.The highest number of accidents were in 2019, followed by 2018.Accidents dropped significantly in 2020-2021,due to COVID-19 lockdowns, with a slight increase in 2022 as aviation operations resumed.

2.The Cessna 208B Grand Caravan experienced  most accidents, followed by Beechcraft 200 Super King Air, Antonov An-2R, De Havilland Canada DHC-6 Twin Otter 300, and Cessna 208 Caravan I.

3.Aircraft classified as ‘written off’ show the highest fatality rates, while accidents with ‘substantial’ damage resulted in relatively few fatalities.This indicates that more severe damage generally corresponds to higher fatalities.

4.Private operators experienced the highest number of accidents, highlighting a potential area for strict safety oversight.  15

Recommendations
1. After COVID-19, more safety measures should be taken as operations resume to normalcy and  also standards to be raised to curb COVID-19 like testing our clients for it to prevent the spread.

2.Purchases should be made on flights with the lowest accidents rates.If need be to purchase flights with high risk extra caution on safety should be taken seriously and thoroughly with very regular maintenace .

3.Prevention of accidents and  impact mitigation to prevent  written off aircraft which causes high fatalites should be taken example better weather monitoring methods and quick emergency response

4.Strict measures on private aircraft should be taken like regular audits ,maintenance and pilot certification since it has the highest accidents.

