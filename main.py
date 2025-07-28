from src.data_loader import data_load, split_data
from src.model import build_model
from src.vizualization import evalute_model, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV

def main():
    #Загрузка данных и разбитие данных
    X, y = data_load('data/parkinsons.data')
    X_train, X_test, y_train, y_test = split_data(X, y)

    #Построение модели
    model = build_model()

    # Параметры для подбора
    param_grid = {
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__n_estimators': [100, 200, 300],
    }

    # Настройка через GridSearchCV
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    #Обучение
    model.fit(X_train, y_train)

    #Предсказание
    y_pred = model.predict(X_test)

    #Оценка и визуализация
    evalute_model(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)

if __name__ == '__main__':
    main()