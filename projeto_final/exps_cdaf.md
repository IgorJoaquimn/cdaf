\* tempos

# **Experimento para a RQ1: Alinhamento e Correlação Temporal**

**Objetivo:** Provar se a variação do sentimento coletivo no chat (WSI e Polaridade) está correlacionada com o desempenho em campo (xT e Momentum) e quantificar o atraso (*delay*) temporal dessa reação.

### **Por que a correlação simples (Pearson minuto a minuto) falha?**

Se você correlacionar o WSI\_{minuto=10} diretamente com o Momentum\_{minuto=10}, o resultado pode ser baixo. Existe um delay natural: a jogada acontece em campo → passa pelo delay do streaming da CazéTV → o torcedor digita → o comentário é processado. O impacto real de um pico de xT no chat só aparece 1 ou 2 minutos depois. Para resolver isso, usa-se a **Função de Correlação Cruzada (CCF)**.

### **Roteiro de Implementação Código:**

1. **Normalização e Suavização das Séries Temporais:**  
   * Como o chat é ruidoso minuto a minuto, aplique uma Média Móvel (Rolling Mean) de 3 minutos nas séries de WSI (ou Polaridade) e de Momentum (ou xT) para capturar a tendência e remover ruídos de alta frequência.  
2. **Cálculo da Correlação Cruzada (Cross-Correlation Function \- CCF):**  
   * Para cada uma das 63 partidas, utilize a função statsmodels.tsa.stattools.ccf ou crie um laço calculando a correlação de Pearson aplicando deslocamentos (*lags*) temporais de k \= \-5 até k \= 5 minutos.  
   * *Fórmula lógica:* CCF(k) \= Corr(Sentimento\_{t}, Campo\_{t-k})  
3. **Agregação dos Resultados da Temporada:**  
   * Calcule o Lag que gera o maior coeficiente de correlação absoluto para cada partida.  
   * Extraia a média e o desvio padrão do coeficiente de correlação máximo entre todas as 63 partidas.  
4. **Teste de Causalidade de Granger (Granger Causality Test):**  
   * Rode o teste (statsmodels.tsa.stattools.grangercausalitytests) para verificar se os valores passados de Momentum e xT trazem informações estatisticamente significativas para prever o WSI atual, fixando um lag máximo de 3 minutos.

### **Resultado esperado:**

* **Gráfico de Correlação Cruzada Médio:** Um gráfico de barras mostrando no eixo X os *lags* (-5 a \+5 minutos) e no eixo Y o coeficiente de correlação médio. O pico esperado deve ocorrer entre os lags \+1 e \+2 (provando o delay da arquibancada digital).  
* **Tabela de Causalidade:** Apresentar os p-valores do Teste de Granger. Se p \< 0.05, você pode afirmar cientificamente: *"O momento tático em campo causa, no sentido de Granger, a variação do sentimento do chat"*.

# 

# **Experimento para a RQ2: Análise de Janela Pré-Evento (Event Study)**

**Objetivo:** Determinar se o chat age como um sensor preditivo/antecipador de eventos críticos (Gols e Cartões) devido à percepção de pressão em campo, ou se ele é puramente reativo.

### **A Lógica do Experimento (Time-to-Event):**

Você precisa isolar o momento de cada evento e criar uma janela temporal padronizada ao redor dele para analisar o comportamento médio do volume de mensagens e do sentimento antes do cronômetro registrar o evento oficial.

### **Roteiro de Implementação Código:**

1. **Filtragem e Janelamento:**  
   * Identifique no dataset match\_minute\_metrics.csv todos os minutos onde a coluna de Gols (ou Cartões) seja igual a 1\.  
   * Para cada gol encontrado, extraia uma janela de 11 minutos: de 5 minutos antes (t-5) até 5 minutos depois (t+5), onde o minuto do gol é o ponto zero (t=0).  
   * *Atenção:* Remova gols que aconteceram antes do minuto 5 de jogo para evitar janelas truncadas.  
2. **Alinhamento e Agregação:**  
   * Crie uma matriz onde cada linha é um gol diferente e as colunas representam o tempo relativo (t-5, t-4, ..., t, ..., t+5).  
   * Preencha essa matriz com os valores de:  
     * Dataset A: Volume Total de Mensagens (normalizado por partida, dividindo pelo volume médio daquela partida específica).  
     * Dataset B: Índice WSI absoluto.  
   * Calcule a **Média** e o **Intervalo de Confiança (95%)** para cada coluna (instante de tempo relativo).  
3. **Teste de Hipótese Estatístico (Validação):**  
   * Defina a "fase basal" (minutos normais de jogo) e a "fase pré-evento" (t-2 e t-1).  
   * Aplique um teste bicaudal (como o **Teste t de Student** para amostras pareadas ou o teste não-paramétrico de **Wilcoxon Signed-Rank**) comparando o volume/WSI da fase pré-evento contra a média basal da partida.

### **Resultado esperado:**

* **Gráfico de Linha de Estudo de Evento (Event Study Plot):** O eixo X vai de \-5 a \+5. O eixo Y mostra o Volume Médio / WSI Médio. Uma linha vertical tracejada marca o ponto t=0 (Momento do Gol).  
  * Shading (sombreamento) ao redor da linha representando o intervalo de confiança.  
* **Análise dos Cenários e Resposta da RQ2:**  
  * *Cenário A (Reativo):* O volume e o WSI se mantêm estáveis de t-5 a t-1, estourando em um pico absurdo apenas em t+1. Conclusão: O chat é puramente reativo.  
  * *Cenário B (Antecipador):* Há uma subida estatisticamente significativa (p \< 0.05) no volume e uma queda/subida no WSI já nos instantes t-2 e t-1. Conclusão: O chat consegue registrar a atmosfera de perigo iminente (pressão de um time, escanteios em sequência, contra-ataques perigosos) antes mesmo do gol de fato acontecer no cronômetro oficial.

### **Resumo das Variáveis Prontas para o seu Prompt do Gerador de Código:**

Quando você for pedir para a IA gerar o código dessas análises usando este contexto, lembre-a de usar as colunas exatas da sua estrutura descrita no README:

* **Série temporal 1 (Chat):** wsi, polarity, volume\_total (derivado da soma das classes).  
* **Série temporal 2 (Campo):** xt\_casa, xt\_fora, momentum.  
* **Eventos discretos:** gols, cartoes.  
* **Frequência:** Minuto a minuto (0 a 90), agrupado por partida (match\_id ou equivalente).

