    Визуализация на генерираните набори от данни:
    


    
![png](output_0_1.png)
    



    
![png](output_0_2.png)
    



    
![png](output_0_3.png)
    



    
![png](output_0_4.png)
    


    
    === Анализ на набор от данни: blobs ===
    
    Резултати от клъстеризацията:
          Алгоритъм  Силует скор  Време (секунди)
             kmeans     0.875647         0.578998
    agglo_euclidean     0.875647         0.006998
    agglo_manhattan     0.875647         0.004001
       agglo_cosine     0.875647         0.005001
    agglo_chebyshev     0.875647         0.004996
    


    
![png](output_0_6.png)
    



    
![png](output_0_7.png)
    



    
![png](output_0_8.png)
    



    
![png](output_0_9.png)
    



    
![png](output_0_10.png)
    



    
![png](output_0_11.png)
    



    
![png](output_0_12.png)
    



    
![png](output_0_13.png)
    



    
![png](output_0_14.png)
    



    
![png](output_0_15.png)
    


    
    === Анализ на набор от данни: moons ===
    
    Резултати от клъстеризацията:
          Алгоритъм  Силует скор  Време (секунди)
             kmeans     0.483590         0.031008
    agglo_euclidean     0.427308         0.017991
    agglo_manhattan     0.445060         0.007000
       agglo_cosine     0.427981         0.004000
    agglo_chebyshev     0.425600         0.003996
    


    
![png](output_0_17.png)
    



    
![png](output_0_18.png)
    



    
![png](output_0_19.png)
    



    
![png](output_0_20.png)
    



    
![png](output_0_21.png)
    



    
![png](output_0_22.png)
    



    
![png](output_0_23.png)
    



    
![png](output_0_24.png)
    



    
![png](output_0_25.png)
    



    
![png](output_0_26.png)
    


    
    === Анализ на набор от данни: anisotropic ===
    
    Резултати от клъстеризацията:
          Алгоритъм  Силует скор  Време (секунди)
             kmeans     0.769653         0.032010
    agglo_euclidean     0.769653         0.013994
    agglo_manhattan     0.769653         0.006998
       agglo_cosine     0.701871         0.006008
    agglo_chebyshev     0.769653         0.003993
    


    
![png](output_0_28.png)
    



    
![png](output_0_29.png)
    



    
![png](output_0_30.png)
    



    
![png](output_0_31.png)
    



    
![png](output_0_32.png)
    



    
![png](output_0_33.png)
    



    
![png](output_0_34.png)
    



    
![png](output_0_35.png)
    



    
![png](output_0_36.png)
    



    
![png](output_0_37.png)
    


    
    === Анализ на набор от данни: varied ===
    
    Резултати от клъстеризацията:
          Алгоритъм  Силует скор  Време (секунди)
             kmeans     0.787119         0.028002
    agglo_euclidean     0.782961         0.017001
    agglo_manhattan     0.782961         0.009001
       agglo_cosine     0.781283         0.003996
    agglo_chebyshev     0.675626         0.005999
    


    
![png](output_0_39.png)
    



    
![png](output_0_40.png)
    



    
![png](output_0_41.png)
    



    
![png](output_0_42.png)
    



    
![png](output_0_43.png)
    



    
![png](output_0_44.png)
    



    
![png](output_0_45.png)
    



    
![png](output_0_46.png)
    



    
![png](output_0_47.png)
    



    
![png](output_0_48.png)
    


    
    Изследване на влиянието на инициализацията на K-means:
    
    Влияние на параметъра n_init върху K-means:
     n_init  mean_silhouette  std_silhouette  mean_runtime
          1         0.875647    1.110223e-16      0.005200
          5         0.875647    1.110223e-16      0.014474
         10         0.875647    1.110223e-16      0.023161
         20         0.875647    1.110223e-16      0.041324
    


    
![png](output_0_50.png)
    


    
    Приложение върху реални данни (лица):
    


    
![png](output_0_52.png)
    


    Брой на истинските лица (хора): 40
    


    
![png](output_0_54.png)
    


    Силует скор за kmeans: 0.1465
    


    
![png](output_0_56.png)
    


    Силует скор за agglo_euclidean: 0.1617
    


    
![png](output_0_58.png)
    


    Силует скор за agglo_manhattan: 0.1320
    


    
![png](output_0_60.png)
    


    Силует скор за agglo_cosine: 0.0910
    


    
![png](output_0_62.png)
    



```python

```
