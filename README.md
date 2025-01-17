# Лабораторная № 2 по курсу "Анализ и разработка данных".

Задание - нужно построить модель классификации, используя [шаблон cooke cutter для Data Science](https://drivendata.github.io/cookiecutter-data-science/) и [DVC](https://dvc.org) для трекинга экспериментов.

## Описание данных

Данные взяты из обучающего соревнования на [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview). По ссылке можно найти описание данных и ноутбуки с примерами кода моделей.

Мы немного разбирали сами данные на практике. Видео можно посмотреть [тут](https://eduhseru.sharepoint.com/:v:/s/AdvancedDataAnalysis2022/EZw_TeFlH5tGgiDp_LO-8JkByc1kg24mZVN9Y4c42MRuPQ?e=f7aCbE) (для доступа нужен аккаунт ВШЭ).

## Требования к реализации

* Нужно выбрать метрику качества и обосновать ее выбор (1 балл).
Метрика RMSE т.к. она хорошо интерпретируем 

* Нужно написать стадии для полного цикла жизни ML модели (4 баллов)
    1. **Препроцессинг.**
    1. **Разделение данных train/val**
    1. **Генерация признаков.** Обратите внимание, что если вы генерируете признаки, которые предполагают обучение на тренировочном датасете (fit), то для валидационного вы должны применять уже обученные трансормации (transform). Так, если бы данные из val к вам пришли из будущего и у вас нет для них правильных ответов. **Данные из val вы никак не используете в обучении/тюнинге параметров/и т.д., только для оценки качества.** Представьте, что данных val у Вас на момент создания модели нет, они придут к вам только в будущем.
    1. **Обучение модели.**
        * Здесь вы можете использовать внутри различные методы оценки качества модели train/test split, k-fold validation и т.д. [Многие из них уже реализованы в scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection). Вы их используете "для себя" чтобы решить какую модель/ модели вы отправите дальше работать с "реальным миром".
        * Нужно имплементировать
            1. Что-то из scikit-learn используя [Scikit-Learn Pipelines...](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline)
            1. [Catboost](https://catboost.ai)
            1. Если захочется, то что-то еще. Больше - можно, меньше - нет.
    1. **Оценка модели** по метрике качества, выбранном в первом пункте на val датасете и [сохранение метрик / графиков](https://dvc.org/doc/start/data-management/metrics-parameters-plots).
    1. **Предсказание (инференс) модели на новых данных.**
    
* Из стейджей выше соберите **один или два пайплайна** (1 балл):
    1. Обучения модели и ее оценки
    1. Инференса модели

* В пайплайнах используется работа с категориальными признаками (2 балла):
    * [CategoricalEncoders](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)
    * [Тюнинг параметров отвечающих за работу с категориями Catboost](https://github.com/catboost/tutorials/blob/master/categorical_features/categorical_features_parameters.ipynb)
* Для [управления данными](https://dvc.org/doc/start/data-management) и [экспериментами](https://dvc.org/doc/start/experiment-management/experiments) использован DVC (2 балла).

Код должен быть написан в функциях и разнесен по модулям шаблона. Если вы изменяете шаблон - напишите об этом комментарий, какая мотивация.
Вы можете использовать ноутбук для разработки, но нужно писать код так, чтобы это потом было легко перенести в функции и разнести по модулям проекта. Пример такой работы мы разбирали на [практике](https://eduhseru.sharepoint.com/:v:/s/AdvancedDataAnalysis2022/EZw_TeFlH5tGgiDp_LO-8JkByc1kg24mZVN9Y4c42MRuPQ?e=f7aCbE).

## Deadlines
* **Hard deadline 23.10.2022 23:59**


## Куда и что отправлять.
1. Выложить весь код и метрики в свой гитхаб
1. Прислать на него ссылку письмом на почту ashimko@hse.ru
    * Тема письма "Лаба № 1. *ФИО* Группа *X*. Анализ и разработка данных.  
    Где *X* это номер вашей группы в соответствии с ведомостью.  
    *ФИО* ваши фамилия, имя, отчество.

Вопросы задавайте в общей группе telegram.  
Удачи!
