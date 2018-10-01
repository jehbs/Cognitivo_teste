# Cognitivo_teste

# Jéssica Barbosa de Souza

# Ciencia de dados

# Predição

# A primeira analise foi no reconhecimento dos dados, consistência, padrões e desvios.
# Após o contato inicial e conhecendo o objetivo da predição, foi decidido utilizar métodos de regressão devido a sua melhor eficiência para as saídas continuas.
# Após a leitura e extração dos pontos de desvio (outliers), foram aplicados diferentes algoritmos: Regressão linear, SVR e XGbootRegression, e comparados seus desempenhos pros dados analisados utilizando FbetaScore quando aplicável. 
# Nesse processo foi detectado um baixo desempenho dos métodos de regressão e decidido interpretar a predição como multiclasses, logo implementou-se métodos de classificação.  XGbootClassificador, Adaboost e SVC foram testados. Foi escolhido o modelo com melhor desempenho na validação do Fscore e em analise manual (validação através de formulas no Excel). Os resultados obtidos foram validados por uma predição com boa acurácia e desempenho. Adaboost foi o selecionado com uma taxa de 67% de acerto. 

#Obs.: rodar o arquivo wineQuality.py