# Прогноз количества заказов для сервиса такси

[HTML](https://github.com/aq2003/Portfolio/blob/main/Taxi%20Service/P12_Portfolio.html) [ipynb](https://github.com/aq2003/Portfolio/blob/main/Taxi%20Service/P12_Portfolio.ipynb)

## Описание проекта

Заказчик - компания такси - собрал исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Надо построить модель для такого предсказания.

Значение метрики *RMSE* на тестовой выборке должно быть не больше 48.

### План проекта

1. Загрузить данные и выполнить ресемплирование по одному часу.
2. Проанализировать данные.
3. Обучить разные модели с различными гиперпараметрами. Тестовая выборка - 10% от исходных данных.
4. Проверить модели на тестовой выборке, сделать выводы.

### Описание данных

Данные лежат в файле `taxi.csv`. Количество заказов находится в столбце '*num_orders*'.

### План работы
1. Подготовка<br/>
1.0. Импорт библиотек<br/>
1.1. Загрузка данных<br/>
1.2. Подготовка данных: исследование на пропуски и выбросы<br/>

2. Анализ<br/>
2.1. Преобразование временного интервала<br/>
2.2. Разложение на тренд, сезонность и остатки для всего периода<br/>
2.3. Исследование внутридневного графика<br/>
2.4. Выводы<br/>

3. Обучение<br/>
3.1. Подготовка признаков<br/>
3.2. Рабиение на выборки<br/>
3.3. Обучение моделей методом кросс-валидации<br/>
3.3.1. LinearRegression<br/>
3.3.2. RandomForrestRegressor<br/>
3.3.3. CatBoostRegressor<br/>

4. Тестирование<br/>
4.1. Тестирование всех моделей на тестовой выборке<br/>
4.2. Выбор лучшей<br/>
4.3. График предсказания для всей выборки<br/>
4.4. Заключение<br/>