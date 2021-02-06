# Active Learning - Comparing Strategies 

## About

Este notebook visa comparar as diversas estratégias de aprendizado ativo encontradas no documento do Burr Settles, disponível em: http://active-learning.net/.

    Obs.: Todas as estratégias foram comparadas utilizando apenas classificadores.

Algumas das estratégias implementadas são:

- Amostra por incerteza
- Amostragem aleatória
- Consulta por comitê
- Aprendizado passivo
- Redução do erro esperado
- Mudança esperada do modelo

## How to use

### virtual env files

``` bash
virtualenv act_learn
source bin/activate
pip install ipython jupyter modAL ...
```

As estruturas do framework seguem o seguinte pipeline:
1. O usuário define quantas instâncias ele deseja através da variável *n_queries* (nota: quanto maior o número de instâncias, maior o custo computacional);

2. É definido um classificador através da função *which_classifier*, sendo os parâmentos:
    - **Classifier:** Define qual o classificador será utilizado no processo;
    - **Parameters:** Os parâmetros a serem utilizados pelo classificador (ex.: número de vizinhos do KNN).
    
3. É definido o dataset através da função *which_dataset* ou *which_oml_dataset*:
    - **dataset (*which_dataset*):** Define o dataset a ser utilizado (ex.: 'iris');
    - **dataset_id (*which_oml_dataset*):** Define pelo id da base do [openML](https://www.openml.org/home) o dataset a ser utilizado (ex.: 61);
    - **n_split:** Define o tamanho das divisões feitas no dataset  (*cross-validation*).
    
4. A função *which_dataset* e *which_oml_dataset* é responsável por retornar:
    - **X_raw:** Características dos dados do conjunto;
    - **y_raw:** Rótulos dos dados do conjunto;
    - **idx_data:** n listas (n = n_split) com a seguinte estrutura: [[train],[test]], nas listas train tendo os ids dos dados de treino e test os ids dos dados de teste. Assim, idx_data[i][j], tal que i = bag e j = treino(0) ou teste(1);

5. Após definir todo o ambiente, uma bateria de funções é executada, sendo essas as estratégias de amostragem do aprendizado ativo junto do dataset e do classificador escolhido.

Cada função de estratégia possui a mesma entrada e saída para padronização do framework, sendo elas:

### Entrada
- **X_raw:** Características dos dados do conjunto;
- **y_raw:**  Rótulos dos dados do conjunto;
- **idx_data:** n listas (n = n_split) de ids do conjunto;
- **idx_bag:** Qual lista se deseja usar (idx_bag de 0 a n_splits-1);
- **classifier:** Qual classificador será utilizado (definido na função *which_classifier*);
- **init_size:**  Tamanho inicial da amostra (toda estratégia parte de um tamanho mínimo aleatório).
- **n_queries:** O número máximo de amostras consultadas (rotuladas) por estratégia. 

### Saída:
- **perfomance_history:** Histórico com acurária do modelo após adição de uma nova amostra ao aprendizado;
- **time_elapsed:** Tempo de execução da estratégia;
- **sample_size:** Quantidade de amostras utilizadas para treino daquele modelo;
- **Strategy:** Estratégia utilizada.


## Libraries

- [modAL](https://modal-python.readthedocs.io/en/latest/);
- [Sklearn](https://scikit-learn.org/stable/index.html);
- [Pandas](https://pandas.pydata.org/);
- [Numpy](https://numpy.org/);
- [openML](https://pypi.org/project/openml/);
- [copy](https://docs.python.org/pt-br/3/library/copy.html);
- [timeit](https://docs.python.org/3/library/timeit.html);
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/);