# Introdução
Hoje em dia, fala-se muito da recessão econômica causada pela pandemia de Covid-19. Isso nos lembra a recessão que ocorreu nos EUA em 2009, também conhecida como crise financeira bancária. Acredita-se que a recessão de 2009 poderia ter sido evitada se os bancos tivessem escolhido cuidadosamente seus pagadores de empréstimos.

Os empréstimos incobráveis na maioria dos casos referem-se a atrasos no pagamento ou inadimplência, que são duas das causas mais comuns de perda de receita bancária e até falência. Atualmente, muitos bancos e fintechs investem amplamente em gestão de risco ou avaliação de crédito para obter informações melhores, baseadas em dados, para tomada de decisões no que diz respeito a concessão de empréstimos.

Neste desafio, desejamos saber de antemão quem é capaz de pagar seus empréstimos e quem não é. O principal objetivo é analisar dados e criar um modelo para descobrir quais clientes são capazes de honrar suas dívidas.

## Metodologia e Desenvolvimento

Dada a introdução acima, concluímos que este projeto se trata da criação de uma solução capaz de classificar os clientes quanto ao seu perfil de pagamento: o cliente é considerado um bom pagador de dívidas ou não?

O trabalho começa com uma análise exploratória de dados que subsidía as tomadas de decisão no decorrer do projeto. Logo de início vemos que as classes target são desequilibradas em número: a base de dados possui muito mais dados sobre bons pagadores do que dados sobre maus pagadores. Em seguida, vemos que existem várias variáveis categóricas na base de dados e que, também, as estatísticas das variáveis numéricas não possibilitam assertivamente a diferenciação entre as duas classes de clientes. Estes achados indicam que a utilização de modelos baseados em árvores de decisão poderia ser uma boa escolha para este problema.

Em seguida, foi feita a escolha do modelo baseline através da avaliação das métricas precision, recall e f1_score de vários modelos classificadores. O XGBClassifier foi o escolhido dentre todos os outros. Após isto, a performance do algoritmo foi melhorada com a execução de feature engineering e eliminação de preditores menos relevantes. Por fim, um ajuste fino dos hiperparâmetros mostrou que é possível melhor a precisão do modelo em troca da piora do recall, e vice-versa. 

## Resultados e Conclusões

Considerando as análises e testes anteriores, no momento, entregaria a seguinte solução: a cominação de um modelo de aprendizado não supervisionado para clusterizar os clientes com um modelo classificador que é capaz de classificar os cliente quanto ao perfil de pagamento e suas probabilidades de pagar ou não suas dívidas.

A solução é capaz de identificar cerca de 92% até 97%, a depender das métricas de negócio, dos bons pagadores e possui uma taxa de acerto de 87%. No que diz respeito aos maus pagadores, este produto identifica cerca de 40% dos maus pagadores, e a taxa de acerto pode variar de 42% até 73% a depender dos objetivos da empresa.

Para escolher corretamente qual conjunto de parâmetros utilizar, precisamos conhecer as respostas para as seguintes pergunta de negócio: o que é mais importante nesta tarefa? Identificar corretamente o maior número possível de bons pagadores ou identificar corretamente o maior número possível de maus pagadores? Ser acertivo na tomada de decisão quanto a classificação do perfil de pagamento? A resposta para estas perguntas pode ser dada pelo time de negócios ou até mesmo pelo cientista de dados, desde que este conheça bem o modelo de negócios com o qual ele trabalha.

De qualquer forma, esta solução entrega um processo automatizado, escalável e que não depende de decisões subjetivas; o que gera economia de tempo, recurso e mão de obra. Além disso, com o uso de uma plataforma cloud, este produto pode ser acessado facilmente por diversas pessoas e aplicações dentro e até fora da empresa; o que abre espaço para criação de novos produtos financeiros.

### Possíveis próximos passos

Utilizar um método estatistico-computacional para inputação de entradas no conjunto de treino que contenham mais informações sobre os clientes que não pagaram suas dívidas, melhorando assim o desbalanceamento das classes presente no dataset.

Utilizar outros algoritmos de regressão para prever os valores faltantes das variáveis numéricas utilizadas no modelo. Em geral, sistemas de AI que utilizam mais de um algoritmo de ML exibem alta performance.

Definir bem os objetivos de negócio de forma a escolher os conjuntos de modelos e parâmetro mais adequados para entregar um produto de dados que melhor soluciona o problema e cumpre com a sua missão dentro da empresa.

