import numpy as np
import matplotlib.pyplot as pl

def feature_plot(importances, X_train, y_train):
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize=(9, 5))
    pl.title("Pesos normalizados dos cinco parâmetros mais influentes", fontsize=16)
    pl.bar(np.arange(5), values, width=0.6, align="center", color='#00A000', \
           label="Peso do parâmetro")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width=0.2, align="center", color='#00A0A0', \
           label="Peso acumulativo do parâmetro")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Peso", fontsize=12)
    pl.xlabel("Parâmetro", fontsize=12)

    pl.legend(loc='upper center')
    pl.tight_layout()
    pl.show()