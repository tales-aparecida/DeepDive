
# coding: utf-8

# Neste relátorio visamos treinar um algoritmo de regressão com base em 50.000 dados com 77 dimensões usando a métrica MAE, erro absoluto médio. Primeiramente diminuiremos as dimensões dos dados com PCA, após, treinaremos diversos regressores para, enfim, realizar um ensemble e reportar as saídas esperadas.

# In[1]:

# Includes
from   sklearn.gaussian_process import GaussianProcessRegressor
from   sklearn.model_selection  import GridSearchCV, train_test_split, KFold, LeaveOneOut
from   sklearn.neural_network   import MLPRegressor
from   sklearn.decomposition    import PCA
from   sklearn.preprocessing    import scale
from   sklearn.linear_model     import LinearRegression
from   sklearn.neighbors        import KNeighborsRegressor
from   sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from   sklearn.metrics          import mean_absolute_error
from   sklearn.utils            import resample, shuffle
from   sklearn.svm              import LinearSVR
import pandas as pd
import numpy  as np
import warnings
import math
import gc

class Stacker(object):
    """
    A transformer applying fitting a predictor `pred` to data in a way
        that will allow a higher-up predictor to build a model utilizing both this 
        and other predictors correctly.

    The fit_transform(self, x, y) of this class will create a column matrix, whose 
        each row contains the prediction of `pred` fitted on other rows than this one. 
        This allows a higher-level predictor to correctly fit a model on this, and other
        column matrices obtained from other lower-level predictors.

    The fit(self, x, y) and transform(self, x_) methods, will fit `pred` on all 
        of `x`, and transform the output of `x_` (which is either `x` or not) using the fitted 
        `pred`.

    Arguments:    
        pred: A lower-level predictor to stack.

        cv_fn: Function taking `x`, and returning a cross-validation object. In `fit_transform`
            th train and test indices of the object will be iterated over. For each iteration, `pred` will
            be fitted to the `x` and `y` with rows corresponding to the
            train indices, and the test indices of the output will be obtained
            by predicting on the corresponding indices of `x`.
    """
    def __init__(self, pred, cv_fn=lambda x: LeaveOneOut().split(x)):
        self._pred, self._cv_fn  = pred, cv_fn

    def fit_transform(self, x, y):
        x_trans = self._train_transform(x, y)

        self.fit(x, y)

        return x_trans

    def fit(self, x, y):
        """
        Same signature as any sklearn transformer.
        """
        self._pred.fit(x, y)

        return self

    def transform(self, x):
        """
        Same signature as any sklearn transformer.
        """
        return self._test_transform(x)

    def _train_transform(self, x, y):
        x_trans = np.nan * np.ones((x.shape[0], 1))

        all_te = set()
        for tr, te in self._cv_fn(x):
            all_te = all_te | set(te)
            x_trans[te, 0] = self._pred.fit(x[tr, :], y[tr]).predict(x[te, :]) 
        if all_te != set(range(x.shape[0])):
            warnings.warn('Not all indices covered by Stacker', sklearn.exceptions.FitFailedWarning)

        return x_trans

    def _test_transform(self, x):
        return self._pred.predict(x)

warnings.filterwarnings("ignore")


# In[2]:

# Lê os dados do arquivo
def readIt():
    from IPython.display import display
    df = pd.io.parsers.read_csv('ex6-train.csv')
    y = df.pop('V78').as_matrix()
    x = scale(df.as_matrix());
    display(df.head())
    return x, y

x, y = readIt()
data_size = len(x)


# # Preprocessamento
# ## PCA

# In[3]:

def pcaIt(x):
    pca = PCA(n_components=0.85)
    pca.fit(x)
    print("Após o PCA temos", pca.n_components_,"componentes")
    return pca.transform(x)
    
x_reduced = pcaIt(x)


# # Análise visual

# In[4]:

def plotIt():
    import matplotlib.pyplot as plt
    
    def plot(i, x, y, ncols, fig):
        if i % ncols == 0:
            fig = plt.figure(figsize=(15,2))
        ax = fig.add_subplot(1, ncols, 1+(i%ncols))
        ax.grid(color='lightgray', linestyle='--', linewidth=1)
        x_, y_ = [[*x] for x in zip(*sorted(zip(x,y)))]
        ax.plot(x_, y_, '.')
        ax.set_title("Componente {}".format(i+1))
        if i % ncols == ncols-1:
            plt.show()
        return fig

    transposed = np.transpose(x)
    transposed_reduced = np.transpose(x_reduced)
    
    fig = None
    ncols = 5
    print("Componentes do dado normalizado")
    for i in range(len(transposed)):
        fig = plot(i, transposed[i], y, ncols, fig)
    plt.show()

    print("Componentes do dado após o PCA")
    for i in range(len(transposed_reduced[:])):
        fig = plot(i, transposed_reduced[i], y, ncols, fig)
    plt.show()
    
#plotIt()


# Analisar as componentes isoladamente pode gerar más interpretações, mas podemos ver que algumas têm comportamentos bem similares, outras, bem comportadas, e ainda há alguns casos, como nas componentes entre 43 e 49, onde os dados variam entre 2 valores.

# # Treinamento

# In[5]:

def getSample(x, y, perc=0.8):
    return shuffle(x, y, n_samples=int(len(x)*perc), random_state=0)

def testIt(regressor, name, params, resultStr, x, y, n_jobs=4):
    gscv = GridSearchCV(regressor, params, n_jobs=n_jobs, scoring='neg_mean_absolute_error', cv=3, verbose=1)
    gscv.fit(x,y)
    regressor = gscv.best_estimator_
    regressor.score = -gscv.best_score_
    paramList = [gscv.best_params_[name] for name in sorted(params.keys())]
    formatParams = [name, -gscv.best_score_] + paramList
    resultStr = "{} >> Menor MAE em um Fold: {:.3f}; " + resultStr
    print(resultStr.format(*formatParams))
    gc.collect()
    return regressor

x = x_reduced
x_smallSample, y_smallSample = getSample(x, y, 500/data_size)
gc.collect()


# ## SVM

# In[6]:

def trainSVM(x, y):
    regressor = LinearSVR(loss='epsilon_insensitive', random_state=0)
    params = {'epsilon':[0, 0.1, 1, 10], 'C':[2**-10, 2**-5, 2**1, 2**5]}
    resultStr = "C: {:.3f}; Eps: {}"
    return testIt(regressor, "SVM", params, resultStr, x, y)

trainSVM(x_smallSample, y_smallSample)


# ## GBM

# In[7]:

def trainGBM(x, y):
    regressor = GradientBoostingRegressor(loss='lad', random_state=0)
    params = {'learning_rate':[0.1, 0.05], 'max_depth':[3, 7], 'n_estimators':[50, 100]}
    resultStr = "L.R.: {}; Max Depth: {}; N Est.: {}"
    return testIt(regressor, "GBM", params, resultStr, x, y)

#trainGBM(x_smallSample, y_smallSample)


# ## RF

# In[8]:

def trainRF(x, y):
    regressor = RandomForestRegressor(criterion='mae', random_state=0, n_jobs=4)
    params = {'n_estimators': [25, 75], 'max_depth': [3, 5]}
    resultStr = "Max Depth: {}; n_estimators: {}"
    return testIt(regressor, "RF", params, resultStr, x, y, n_jobs=1)

#trainRF(x_smallSample, y_smallSample)


# ## RN

# In[9]:

def trainRN(x, y):
    regressor = MLPRegressor(random_state=0)
    params = {'alpha':[1e-5, 1, 1e+5], 
              'hidden_layer_sizes':[(30), (70,)]}
    resultStr = "alpha: {}; Tam.Camadas: {}"
    return testIt(regressor, "RN", params, resultStr, x, y)

#trainRN(x_smallSample, y_smallSample)


# ## KNN

# In[10]:

def trainKNN(x, y):
    regressor = KNeighborsRegressor(p=1)
    params = { 'n_neighbors':[1, 5, 11, 15, 21, 25], 'weights':['distance', 'uniform']}
    resultStr = "n_neighbors: {}; weights: {}"
    return testIt(regressor, "KNN", params, resultStr, x, y)

trainKNN(x_smallSample, y_smallSample)


# ## Gaussian

# In[11]:

def trainGaussian(x, y): 
    regressor = GaussianProcessRegressor(copy_X_train=False, random_state=0)
    params = { 'alpha':[1e-7, 1, 1e+7]}
    resultStr = "alpha: {}"
    return testIt(regressor, "Gaussian", params, resultStr, x, y)

trainGaussian(x_smallSample, y_smallSample)


# # Ensemble

# In[12]:

# Os códigos para ensemble são grandemente inspirados de uma resposta
# creditada à Shahar Azulay & Ami Tavory. Acessado em 17/06/2017
# url: https://stackoverflow.com/a/35170149/3171285

# Tamanhos arbitrariamente escolhidos conforme capacidade da máquina
sample_size = int(data_size)
train_size = int(sample_size*0.8)
test_size = sample_size - train_size

# Separa treino e teste para checar acurácia da união dos 6 métodos
i_tr, i_ts = train_test_split(
    list(range(data_size)), 
    train_size=train_size, 
    test_size=test_size)
x_tr, x_ts, y_tr, y_ts = x[i_tr], x[i_ts], y[i_tr], y[i_ts]


# In[25]:

# Gera objetos dos regressores com melhores hiperparâmetros para um subsample do treino
regressors = []


# In[26]:

regressors.append(trainSVM(*getSample(x_tr, y_tr))) 


# In[27]:

regressors.append(trainGBM(*getSample(x_tr, y_tr, 0.2))) 


# In[28]:

regressors.append(RandomForestRegressor(criterion='mae', 
                                        random_state=0, 
                                        n_jobs=4, 
                                        n_estimators=75, 
                                        max_depth=5))
#regressors.append(trainRF(*getSample(x_tr, y_tr, 0.2)))


# In[29]:

regressors.append(trainRN(*getSample(x_tr, y_tr, 0.2))) 


# In[ ]:

regressors.append(trainKNN(*getSample(x_tr, y_tr))) 


# In[ ]:

regressors.append(trainGaussian(*getSample(x_tr, y_tr, 0.2)))


# In[ ]:

n_regressors = len(regressors)
print("Qnt. de Regressores: ", n_regressors)
# Inicializa o vetor de predições para y
y_pred = np.zeros((data_size,n_regressors))

# As predições com dados de treino excluirão o dado a ser previsto do treino
cv_fn = lambda x: KFold().split(x)

# Gera a predição para cada regressor
for i in range(n_regressors):
    print(regressors[i])
    stacker = Stacker(regressors[i],cv_fn=cv_fn)
    y_pred[i_tr, i] = stacker.fit_transform(x_tr, y_tr)[:,0]
    y_pred[i_ts, i] = stacker.fit(x_ts, y_ts).transform(x_ts)


# In[ ]:

# Gera um regressor para prever qual a melhor previsão (XD)
u = MLPRegressor(hidden_layer_sizes=(50, ), max_iter=500, random_state=0).fit(y_pred[i_tr, :], y_tr)
w = LinearRegression().fit(y_pred[i_tr, :], y_tr)

# Calcula a pontuação deste regressor 
print("MLP: {:.3f}".format(mean_absolute_error(y_ts, u.predict(y_pred[i_ts, :]))))
print("Linear: {:.3f}".format(mean_absolute_error(y_ts, w.predict(y_pred[i_ts, :]))))


#     Obs.: O resultado esperado pelos dados de teste estão no arquivo ex6-result.csv  
#           São criadas funções para trechos triviais para otimizar a memória disponível.

# In[ ]:



