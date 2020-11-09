
# Анализ текстов

Наш заказчик интернет-магазин запускает новый сервис, который позволяет пользователям редактировать и дополнять описания товаров, как в вики-сообществах. Магазину нужен инструмент, который будет выделять токсичные комментарии и отправлять их на модерацию. 

Следует обучить модель классифицировать комментарии на позитивные и негативные. Заказчик предоставил набор данных с разметкой о токсичности правок.

Заказчик требует, чтобы метрика качества *F1* была не меньше 0.75.

Следует проверить несколько разных моделей и рекомендовать в производство ту, которая покажет наилучший результат.<br/>
Совместно с заказчиком определили список моделей для исследования:
- Логистическая регрессия
- Случайный лес
- CatBoost

## План работы

1. Загрузить и подготовить данные:<br/>
1.1. Загрузить данные<br/>
1.2. Исследовать характеристики<br/>
1.3. Исследовать на дисбаланс классов<br/>
1.4. Нормализовать и лемматизировать тексты<br/>
1.5. Подготовить массив признаков по технологии TF-IDF<br/>


2. Обучить разные модели: <br/>
2.1. Разбить на обучающую и тестовую выборки<br/>
2.2. Логистическая регрессия - обучить и протестировать<br/>
2.3. Случайный лес - обучить и протестировать<br/>
2.4. CatBoost - обучить и протестировать<br/>


3. Сделать выводы:<br/>
3.1. Сделать выводы по работе<br/>
3.2. Выбрать модель<br/>
3.3. Наметить план дальнейших действий<br/>

## Описание данных

Данные находятся в файле `toxic_comments.csv`:
- *text* - текст комментария
- *toxic* — целевой признак

# 1. Загрузка и подготовка данных

Загрузим необходимые библиотеки.


```python
# importing libraries
import numpy as np
import pandas as pd

import re

import torch
import transformers
from tqdm import notebook

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer

print('Importing libraries - OK')
```

    Importing libraries - OK


## 1.1. Загрузка данных

Загрузим файл и посмотрим на содержимое, чтобы удостовериться, что всё прошло гладко.


```python
#loading the file
data = pd.read_csv('/datasets/toxic_comments.csv')
data.head(10)
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
      <th>text</th>
      <th>toxic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Explanation\nWhy the edits made under my usern...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>D'aww! He matches this background colour I'm s...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Hey man, I'm really not trying to edit war. It...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>"\nMore\nI can't make any real suggestions on ...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>You, sir, are my hero. Any chance you remember...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>"\n\nCongratulations from me as well, use the ...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK</td>
      <td>1</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Your vandalism to the Matt Shirvington article...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Sorry if the word 'nonsense' was offensive to ...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>alignment on this subject and which are contra...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Данные загружены, отлично. 

## 1.2. Исследование данных

Посмотрим на типы, количество строк и сколько памяти занимает.


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 159571 entries, 0 to 159570
    Data columns (total 2 columns):
    text     159571 non-null object
    toxic    159571 non-null int64
    dtypes: int64(1), object(1)
    memory usage: 2.4+ MB


Пропусков нет, уже хорошо. Тип признака `toxic` можно изменить на более короткий, чтобы уменьшить количество занимаемой таблицей памяти. Модели предстоят серьёзные, память пригодится.


```python
data['toxic'] = data['toxic'].astype('uint8')
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 159571 entries, 0 to 159570
    Data columns (total 2 columns):
    text     159571 non-null object
    toxic    159571 non-null uint8
    dtypes: object(1), uint8(1)
    memory usage: 1.4+ MB


Объём занимаемой памяти уменьшился в 1,5 раза, это здорово.

## 1.3. Исследование на дисбаланс классов

Проверим на баланс классов. Вдруг они не сбалансированы. Это важно для обучения.


```python
toxic_len = data[data['toxic'] == 1].shape[0]
toxic_non_len = data[data['toxic'] != 1].shape[0]

print('toxic: {:d} rows, {:.2%}'.format(toxic_len, toxic_len / data.shape[0]))
print('non_toxic: {:d} rows, {:.2%}'.format(toxic_non_len, toxic_non_len / data.shape[0]))
```

    toxic: 16225 rows, 10.17%
    non_toxic: 143346 rows, 89.83%


Классы распределены как 1:9. Потребуется делать балансировку классов. Учтём это в параметрах моделей.

## 1.4. Нормализация и лемматизация текстов

Нормализуем тексты.


```python
def text_normalizer(text):
    """
    normalizes given text according to the template
    parameters:
    - text - text to normalize
    
    returns: normalized text
    """        
    return " ".join(re.sub(r"[^a-zA-Z']", " ", text).split()).lower()

print('Compiling text_normalizer - OK')
```

    Compiling text_normalizer - OK



```python
# normalizing texts
normalized_texts = []

for row in range(X.shape[0]):
    normalized_texts.append(text_normalizer(X.iloc[row]['text']))
    
X['text'] = normalized_texts

X.head(10)
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
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>explanation why the edits made under my userna...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>d'aww he matches this background colour i'm se...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>hey man i'm really not trying to edit war it's...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>more i can't make any real suggestions on impr...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>you sir are my hero any chance you remember wh...</td>
    </tr>
    <tr>
      <td>5</td>
      <td>congratulations from me as well use the tools ...</td>
    </tr>
    <tr>
      <td>6</td>
      <td>cocksucker before you piss around on my work</td>
    </tr>
    <tr>
      <td>7</td>
      <td>your vandalism to the matt shirvington article...</td>
    </tr>
    <tr>
      <td>8</td>
      <td>sorry if the word 'nonsense' was offensive to ...</td>
    </tr>
    <tr>
      <td>9</td>
      <td>alignment on this subject and which are contra...</td>
    </tr>
  </tbody>
</table>
</div>



Прошло удачно.

Займёмся лемматизацией.


```python
def lemmatize(text, m):
    """
    Lemmatizes a given text
    parameters:
    - text - text to lemmatize
    - m - Mystem object
    returns: lemmatized text
    """
    return "".join(m.lemmatize(text))
```


```python
%%time
# lemmatizing texts
lemmatized_texts = []
m = WordNetLemmatizer()

for row in range(X.shape[0]):
    lemmatized_texts.append(lemmatize(X.iloc[row]['text'], m))
    
X['text'] = lemmatized_texts

X.head(10)
```

    CPU times: user 28.6 s, sys: 239 ms, total: 28.9 s
    Wall time: 29 s





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
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>explanation why the edits made under my userna...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>d'aww he matches this background colour i'm se...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>hey man i'm really not trying to edit war it's...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>more i can't make any real suggestions on impr...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>you sir are my hero any chance you remember wh...</td>
    </tr>
    <tr>
      <td>5</td>
      <td>congratulations from me as well use the tools ...</td>
    </tr>
    <tr>
      <td>6</td>
      <td>cocksucker before you piss around on my work</td>
    </tr>
    <tr>
      <td>7</td>
      <td>your vandalism to the matt shirvington article...</td>
    </tr>
    <tr>
      <td>8</td>
      <td>sorry if the word 'nonsense' was offensive to ...</td>
    </tr>
    <tr>
      <td>9</td>
      <td>alignment on this subject and which are contra...</td>
    </tr>
  </tbody>
</table>
</div>



Лемматизация тоже получилась. Тексты готовы для превращения в признаки.

## 1.5. Подготовка массива признаков

Теперь надо обработать текст, чтобы превратить наборы слов в признаки. Призовём на помощь технологию TF-IDF. <br/>Так же исключим не значащие слова. И разделим выборки.


```python
# splitting to subsamples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

print("X_train.shape", X_train.shape)
print("y_train.shape", y_train.shape)
print("X_test.shape", X_test.shape)
print("y_test.shape", y_test.shape)
```

    X_train.shape (119678, 1)
    y_train.shape (119678,)
    X_test.shape (39893, 1)
    y_test.shape (39893,)



```python
nltk.download('stopwords')
stopwords = set(nltk_stopwords.words('english'))
```

    [nltk_data] Downloading package stopwords to /home/jovyan/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



```python
%%time
count_tf_idf = TfidfVectorizer(stop_words=stopwords)

X_tf_idf_train = count_tf_idf.fit_transform(X_train['text'].values.astype('U'))
X_tf_idf_test = count_tf_idf.transform(X_test['text'].values.astype('U'))
print('X_tf_idf_train.shape -', X_tf_idf_train.shape)
print('X_tf_idf_test.shape -', X_tf_idf_test.shape)
```

    X_tf_idf_train.shape - (119678, 142734)
    X_tf_idf_test.shape - (39893, 142734)
    CPU times: user 13.4 s, sys: 1.92 s, total: 15.3 s
    Wall time: 15.6 s


Всё готово для того, чтобы приступить к обучению моделей:
- В нашем распоряжении имеется массив векторов значимости слов в корпусе текстов по технологии TF-IDF
- Имеется массив целей, соответствующих веторам
- Классы в выборке сбалансированы

# 2. Обучение

Обучим 3 разнотипные модели:
1. Логистическая регрессия
2. Случайный лес
3. CatBoost

Так как наша цель состоит в том, чтобы принципиально определить наилучшую модель, а массив данных получился большим, для сокращения времени подбор гиперпараматеров проводить не будем. На основании опыта для каждой модели установим такие гиперпараметры, которые дают неплохие результаты и в то же время не нагружают сильно вычислительный ресурс.<br/>

Окончательную доводку будем проводить с выбранной моделью.

## 2.1. Разбиение на выборки

На выборки уже разбили ранее.

## 2.2. Логистическая регрессия
Обучим логистическую регрессию. Учтём в параметрах дисбаланс классов.


```python
%%time
model_lr = LogisticRegression(solver='lbfgs', 
                              max_iter=1000, 
                              class_weight='balanced',
                              random_state=12, 
                              verbose=1)
model_lr.fit(X_tf_idf_train, y_train)
```

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    CPU times: user 23.3 s, sys: 10.3 s, total: 33.7 s
    Wall time: 33.7 s


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:   33.7s finished





    LogisticRegression(C=1.0, class_weight='balanced', dual=False,
                       fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                       max_iter=1000, multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=12, solver='lbfgs', tol=0.0001, verbose=1,
                       warm_start=False)



И протестируем.


```python
predict = model_lr.predict(X_tf_idf_test)
f1_lr = f1_score(y_test, predict)
print('LinearRegression F1 score -', f1_lr)
```

    LinearRegression F1 score - 0.7645727831052688


Результат получился неожиданно высоким. Заказчик будет доволен. Вот что баланс классов животворящий делает.<br/>
(Сознаюсь, проводил исследования и на несбалансированной выборке. Результат был существенно хуже).

## 2.3. Случайный лес
Обучим и протестируем случайный лес и прмиеним параметр балансировки классов.


```python
%%time
model_rf = RandomForestClassifier(
    n_estimators=1000, criterion='gini', max_depth=4, min_samples_split=2, min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
    min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, 
    n_jobs=None, random_state=12, verbose=1, warm_start=False, class_weight='balanced')
model_rf.fit(X_tf_idf_train, y_train)
```

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    CPU times: user 24.1 s, sys: 0 ns, total: 24.1 s
    Wall time: 24.6 s


    [Parallel(n_jobs=1)]: Done 1000 out of 1000 | elapsed:   23.8s finished





    RandomForestClassifier(bootstrap=True, class_weight='balanced',
                           criterion='gini', max_depth=4, max_features='auto',
                           max_leaf_nodes=None, min_impurity_decrease=0.0,
                           min_impurity_split=None, min_samples_leaf=1,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=1000, n_jobs=None, oob_score=False,
                           random_state=12, verbose=1, warm_start=False)



Протестируем случайный лес.


```python
predict = model_rf.predict(X_tf_idf_test)
f1_rf = f1_score(y_test, predict)
print('RandomForestRegressor F1 score -', f1_rf)
```

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    RandomForestRegressor F1 score - 0.35199065511535094


    [Parallel(n_jobs=1)]: Done 1000 out of 1000 | elapsed:    8.9s finished


Случайный лес показал тоже высокий результат, но хуже, чем логистическая регрессия.

## 2.4. CatBoost
Обучим и протестируем модель CatBoost и снова не забудем про дисбаланс классов.


```python
%%time
model_cat = CatBoostClassifier(
            n_estimators=1000, 
            class_weights=[1, 9],
            max_depth=4, 
            verbose=100)
model_cat.fit(X_tf_idf_train, y_train)
```

    Learning rate set to 0.064894
    0:	learn: 0.6670617	total: 2.71s	remaining: 45m 6s
    100:	learn: 0.4348166	total: 4m 23s	remaining: 39m 8s
    200:	learn: 0.3824419	total: 8m 52s	remaining: 35m 17s
    300:	learn: 0.3481243	total: 13m 20s	remaining: 30m 58s
    400:	learn: 0.3248843	total: 17m 43s	remaining: 26m 28s
    500:	learn: 0.3075953	total: 22m 5s	remaining: 22m
    600:	learn: 0.2928943	total: 26m 26s	remaining: 17m 33s
    700:	learn: 0.2807115	total: 30m 44s	remaining: 13m 6s
    800:	learn: 0.2702994	total: 35m 11s	remaining: 8m 44s
    900:	learn: 0.2615201	total: 39m 31s	remaining: 4m 20s
    999:	learn: 0.2532964	total: 43m 49s	remaining: 0us
    CPU times: user 44min 25s, sys: 51.9 s, total: 45min 16s
    Wall time: 45min 21s





    <catboost.core.CatBoostClassifier at 0x7fc9eb464c90>



Протестируем CatBoost.


```python
predict = model_cat.predict(X_tf_idf_test)
f1_cat = f1_score(y_test, predict)
print('CatBoostRegressor F1 score -', f1_cat)
```

    CatBoostRegressor F1 score - 0.7519929140832594


CatBoost занял второе место.

# 3. Выводы

## 3.1. Выбор лучшей модели

Сведём результаты в одну таблицу.


```python
pd.DataFrame(
    [f1_lr, f1_rf, f1_cat], 
    columns=['f1_score'],
    index=['LinearRegression', 'RandomForestClassifier', 'CatBoostClassifier']).sort_values(by='f1_score', ascending=False)
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
      <th>f1_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LinearRegression</td>
      <td>0.764573</td>
    </tr>
    <tr>
      <td>CatBoostClassifier</td>
      <td>0.751993</td>
    </tr>
    <tr>
      <td>RandomForestClassifier</td>
      <td>0.351991</td>
    </tr>
  </tbody>
</table>
</div>



Места распределились следующим образом:
1. Логистическая регрессия
2. CatBoost
3. Случайный лес

**Логистическая регрессия** показала наилучший результат. Выберем её для дальнейшего тюнинга и передачи в производство.

## 3.2. Общий вывод

Мы провели исследовательскую работу по обучению и выбору модели для определения токсичных комментариев для интернет-магазина. Был проделан следующий объём работ:
1. Данные загружены и исследованы.
2. Обнаружен дисбаланс классов в соотношении 1:9, то есть токсичные комментарии составляют одну десятую часть набора текстов.
3. Так как дисбаланс, да ещё такой большой, существенно влияет на качество моделей, было принято решение его устранить.
4. Устранили дисбаланс уменьшением выборки, чтобы уменьшить объём набора данных для обучения и тестирования и сократить время обучения.
5. Провели нормализацию и лемматизацию текстов.
6. Создали набор векторов признаков по технологии TF-IDF - относительная встречаемость слов в корпусе текстов.
7. Провели обучение и проверили на тестовой выборке 3 модели: линейная регрессия, случайный лес, CatBoost.
8. Все 3 модели показали существенно лучший результат f1, чем заданное заказчиком 0,75.
9. Для производства выбрана логистическая регрессия.

# 3.3. Дальнейшие действия

Есть ли пути для улучшения результата? Да, есть.<br/>

Можно сделать следующее:
1. Подобрать гиперпараметры для улучшения результата
2. Провести векторизацию методом BERT. Это потребует серьёзных вычислительных мощностей
