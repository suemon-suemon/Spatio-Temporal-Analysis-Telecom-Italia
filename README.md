# Spatio-Temporal-Analysis-Telecom-Italia

## Model Description

| Model Name           | Type            | Data Type      |                30days-10mins                |         60days-10mins         |           60days-1hour           |
| -------------------- | --------------- | -------------- | :-----------------------------------------: | :----------------------------:| :------------------------------: |
|                      |                 |                |                 MAE\|  RMSE                 |          MAE\|  RMSE          |           MAE\|  RMSE           |
| HI                   | Time Series     | Time Series    |                 30.606\| -                  |     12.560\| ***17.628***     |  90.631\| ***128.092***  |
| ARIMA                | Time Series     | Time Series    |                 28.024\| -                  |             -\| -             |              -\| -              |
| Theta                | Time Series     | Time Series    |                 27.230\| -                  |             -\| -             |              -\| -              |
| LSTM                 | Time Series +   | Sliding Window |               39.780\| 88.138               |             -\| -             |        227.606\| 526.731        |
| CNN-LSTM             | Spatio-temporal | Sliding Window |                    -\| -                    |             -\| -             |              -\| -              |
| STN                  | Spatio-temporal | Sliding Window |             25.843\| ***50.051***           |        12.014\| 22.443        |         96.799\| 173.383        |
| Vanilla TransformerE | Time Series +   | Sliding Window |               28.745\| 54.207               |             -\| -             |              -\| -              |
| ***Informer***       | Time Series +   | Sliding Window |               26.003\| 53.063               | ***11.568*** \| 24.009 | ***74.6274*** \| 171.131 |
| StTran               | Spatio-temporal | Sliding Window |               27.875\| 56.149               |        11.896\| 24.597        |              -\| -              |
| ***ViT***            | Spatio-temporal | Sliding Window |            ***25.414*** \| 53.514           |             -\| -             |              -\| -              |
| -                    | -               | -              |                      -                      |               -               |                -                |
| MLP                  | Time Series     | Full grid      |              26.562\| 52.882                |             -\| -             |              -\| -              |
| ST-DenseNet          | Spatio-temporal | Full grid      |              46.952\| 154.664               |        33.475\| 75.603        |        203.403\| 517.890        |
| MVSTGN               | Spatio-temporal | Full grid      |              47.848\| 147.858               |        16.973\| 56.853        |        156.780\| 408.457        |
| STGCN                | Spatio-temporal | Full grid      |               28.662\| 58.667               |        14.742\| 40.747        |        255.056\| 480.541        |
| ***ViT-Pyramid***    | Spatio-temporal | Full grid      |               27.115\| 53.869               |         24.239\| 41.66        |                                 |
| ***ViT***            | Spatio-temporal | Full grid      |           ***25.84***\|***51.563***         |   ***11.649***\| ***20.96***  | ***71.639***\| ***164.173***   |

# Time Series Datasets

| Subject | Name      | Temporal Resolution | Statistics           | Description             | Type   |
|----------|-----------|---------------------|----------------------|--------------------------|--------|
| Traffic  | METR-LA   | 5 minutes            | (207, 34,272)         | traffic speed data       | Tensor |
| Traffic  | PEMS-BAY  | 5 minutes            | (326, 52,116)         | Traffic speed and flow   | Tensor |
| Traffic  | PEMS-2    | 5 minutes            | (325, 61,056)         | Traffic speeds           | Tensor |
| Traffic  | LargeST   | 5 minutes            | (8600, 525,888)       | Traffic sensors          | Tensor |
| Traffic  | TrafficBJ | 5 minutes            | (3126, 21,600)        | Traffic speeds           | Tensor |
| Traffic  | TaxiBJ    | 30 minutes           | 4*(7220, 2, 32, 32)   | Crowd flows               | Tensor |
| Traffic  | JONAS-NYC | 30 minutes           | 2*(4800, 16, 8, 2)    | Demand & Supply          | Tensor |
| Traffic  | JONAS-DC  | 1 hour               | 2*(2400, 9, 12, 2), 3* | Demand & Supply          | Tensor |
| Traffic  | COVID-CHI | 2 hours              | (4800, 51, 10)        | Demand & Supply          | Tensor |
| Traffic  | COVID-US  | 1 hour               | (4800, 51, 10)        | Travel purpose            | Tensor |
| Traffic  | BikeNYC   | (D*24,16,8,2)       | N=#of days            | Bike trip records         | Tensor |
| Traffic  | TaxiNYC   | 22,394,490           | trip records           | Taxi trip records         | Record |
| Finance  | M4        | 1 hour ~ 1 year      | 100,000 time series   | Time series               | Series |
| Finance  | M5        | 1 day                | (30490, 1947), (30490, | Walmart sales forecast   | Series |
| Finance  | NASDAQ    | 1 minute              | (104, 391*191), (104,  | Stock prices             | Series |
| Finance  | Stock     | 1 day                | 8049*(D, 7)           | NASDAQ stock prices       | Series |
| Finance  | stocknet- | 1 day                | (88, 731, 5)          | Stock price movement      | Series |
| Finance  | CSI300    | 1 day                |                       | Stock market index        | Series |
| Telecom  | Milan&T   | 15 minutes            | (T, 100, 100, 5)      | CDRs                      | Tensor |
| Health   | ECG       | 2 ms, 10 ms          | Sample rate 100Hz, 500Hz| ECG dataset               | Series |
| Health   | MIMIC-III | various               |                       | Clinical data             | Series |
| Health   | MIT-BIH   | 2.78 ms               | 48 records             | ECG dataset               | Series |
| Health   | PTB       | 1 ms                  | 549 records            | ECG dataset               | Series |
| Weather  | Shifts    |                      | (3129592, 129)        | Weather prediction        | Series |
| Energy   | Electricity| 15 minutes           | (370,140256)         | Electricity consumption   | Series |
