from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evalute_model(y_true, y_pred):
    print('Классификационный отчет:\n')
    print(classification_report(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['parkinson', 'healthy'],
                yticklabels=['parkinson', 'healthy'])
    plt.xlabel('Предсказано')
    plt.title('Матрица ошибок')
    plt.show()