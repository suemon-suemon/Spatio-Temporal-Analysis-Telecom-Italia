# Spatio-Temporal-Analysis-Telecom-Italia

## Model Description

| Model Name           | Type            | Data Type      |         30days-10mins         |         60days-10mins         |           60days-1hour           |
| -------------------- | --------------- | -------------- | :----------------------------:| :----------------------------:| :------------------------------: |
|                      |                 |                |          MAE\|  RMSE          |          MAE\|  RMSE          |           MAE\|  RMSE            |
| HI                   | Time Series     | Time Series    |           30.606\| -          | 12.560\| ***17.628*** |  90.631\| ***128.092***  |
| ARIMA                | Time Series     | Time Series    |           28.024\| -          |             -\| -             |              -\| -               |
| Theta                | Time Series     | Time Series    |           27.230\| -          |             -\| -             |              -\| -               |
| LSTM                 | Time Series +   | Sliding Window |        39.780\| 88.138        |             -\| -             |        227.606\| 526.731         |
| CNN-LSTM             | Spatio-temporal | Sliding Window |             -\| -             |             -\| -             |              -\| -               |
| STN                  | Spatio-temporal | Sliding Window | 25.843\| ***50.051***         |        12.014\| 22.443        |         96.799\| 173.383         |
| Vanilla TransformerE | Time Series +   | Sliding Window |        28.745\| 54.207        |             -\| -             |              -\| -               |
| Informer             | Time Series +   | Sliding Window |        26.003\| 53.063        | ***11.568*** \| 24.009        | ***74.6274*** \| 171.131         |
| StTran               | Spatio-temporal | Sliding Window |        27.875\| 56.149        |        11.896\| 24.597        |              -\| -               |
| ViT                  | Spatio-temporal | Sliding Window | ***25.414*** \| 53.514        |             -\| -             |              -\| -               |
| - | - | - | - | - | - |
| MLP                  | Time Series     | Full grid      | ***26.562***\| ***52.882***   |         -\| -                 |        -\| -                     |
| ST-DenseNet          | Spatio-temporal | Full grid      |        46.952\| 154.664       |        33.475\| 75.603        |        203.403\| 517.890         |
| MVSTGN               | Spatio-temporal | Full grid      |        47.848\| 147.858       |        16.973\| 56.853        |        156.780\| 408.457         |
| STGCN                | Spatio-temporal | Full grid      |        28.662\| 58.667        |        14.742\| 40.747        |        255.056\| 480.541         |
| ViT-Pyramid          | Spatio-temporal | Full grid      |        27.115\| 53.869        |    24.239\| 41.66             |                                  |
| ViT                  | Spatio-temporal | Full grid      |         -\| -                 |     -\| -                     |                                  |
