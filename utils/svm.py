import csv
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from joblib import Parallel, delayed, dump
import numpy as np
import time


class SVMClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', C=1)

    def grid_search(self, X_train, y_train, param_grid, cv=5, n_jobs=-1):
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, n_jobs=n_jobs)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, data_type, train_type='no_gan', dir_='results/svm'):
        model_filename = f'models/svm_without_cross_val_{data_type}_model.pkl'
        init_time = time.time()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        cur_time = time.time()
        time_taken = cur_time - init_time
        print(f"Time taken: {time_taken:.4f}s")

        if data_type == 'iris':
            self.plot_decision_boundary(X_train, y_train)

        if train_type != 'no_gan':
            dir_ = 'results/svm_gan'
            model_filename = f'models/svm_gan_without_cross_val_{data_type}_model.pkl'

        # save true labels and predictions to a CSV file
        with open(f'{dir_}/{data_type}_predictions.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['True Label', 'Predicted Label'])

            for true_label, pred_label in zip(y_test, y_pred):
                writer.writerow([true_label, pred_label])

        # evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Accuracy: %.2f%%" % (accuracy * 100))
        print(f"F1 Score: %.2f%%" % (f1 * 100))

        # save the trained model to a file
        dump(self.model, model_filename)

        return y_pred

    def train_and_validate(self, X_train, y_train, data_type, train_type='no_gan',
                           save_directory='images/svm_cross_validation'):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        def train_fold(train_index, val_index):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            self.model.fit(X_train_fold, y_train_fold)

            y_pred = self.model.predict(X_val_fold)
            return accuracy_score(y_val_fold, y_pred), f1_score(y_val_fold, y_pred, average='macro')

        init_time = time.time()
        results = Parallel(n_jobs=-1)(
            delayed(train_fold)(train_index, val_index) for train_index, val_index in skf.split(X_train, y_train)
        )

        accuracies, f1_scores = zip(*results)
        cur_time = time.time()
        time_taken = cur_time - init_time
        print(f"Time taken: {time_taken:.4f}s")

        # save the trained model to a file
        model_filename = f'models/svm_with_cross_val_{data_type}_model.pkl'
        dump(self.model, model_filename)

        print(f"Mean Accuracy: %.2f%%" % (np.mean(accuracies) * 100))
        print(f"Mean F1 Score: %.2f%%" % (np.mean(f1_scores) * 100))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, 11), accuracies, marker='o', color='red')
        plt.title('Accuracy Progression')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')

        for i, acc in enumerate(accuracies):
            plt.text(i + 1, acc, f'{acc:.2f}', ha='right', va='bottom', fontsize=8)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, 11), f1_scores, marker='o', color='green')
        plt.title('F1 Score Progression')
        plt.xlabel('Fold')
        plt.ylabel('F1 Score')

        for i, f1 in enumerate(f1_scores):
            plt.text(i + 1, f1, f'{f1:.2f}', ha='right', va='bottom', fontsize=8)

        plt.tight_layout()

        if train_type != 'no_gan':
            save_directory = 'images/svm_gan_cross_validation'

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        save_path = os.path.join(save_directory, f'{data_type}_evaluation.png')
        plt.savefig(save_path)
        # plt.show()

    # function to plot decision boundary for Iris dataset
    def plot_decision_boundary(self, X, y):
        X = X[:, :2]

        def make_meshgrid(x, y, h=.02):
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            return xx, yy

        def plot_contours(ax, clf, xx, yy, **params):
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            out = ax.contourf(xx, yy, Z, **params)
            return out

        clf = self.model.fit(X, y)

        fig, ax = plt.subplots()
        # title for the plots
        title = 'Decision Boundary of Iris Dataset'
        # set-up grid for plotting.
        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = make_meshgrid(X0, X1)

        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_ylabel('Feature 1')
        ax.set_xlabel('Feature2')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

        save_directory = 'images/svm_result_samples'

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        save_path = os.path.join(save_directory, f'iris_decision_boundary.png')
        plt.savefig(save_path)
        # plt.show()
