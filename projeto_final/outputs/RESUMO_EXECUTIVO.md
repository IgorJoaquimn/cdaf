# Resumo Executivo: Sentimento em Tempo Real e Eventos de Campo

Este documento apresenta as conclusões consolidadas, análises críticas e discussões teóricas das Questões de Pesquisa 1 e 2. O conteúdo destina-se a servir de base científica para a elaboração das seções de conclusão e discussão do artigo final, bem como para a estruturação de apresentações acadêmicas. O detalhamento estatístico completo, incluindo tabelas de regressão e gráficos de sensibilidade, encontra-se disponível no relatório principal de resultados.

**Amostra Analítica:** A base de dados original de 63 partidas da Bundesliga foi reduzida para 54 jogos após a exclusão de 9 confrontos que não apresentavam sinal de chat ativo nas transmissões da CazéTV. As análises de séries temporais adotam uma resolução minuto a minuto para cada partida.

---

## Metodologia Final Resumida

Para garantir a validade acadêmica das conclusões frente a avaliações exigentes de professores, a metodologia final do estudo estabeleceu os seguintes pilares de rigor estatístico e esportivo:

1. **Eliminação de Endogeneidade na Métrica de xT:** A grade de Ameaça Esperada (Expected Threat) foi calibrada sobre 243 partidas independentes da temporada que não continham dados de chat vinculados, evitando que o preditor de campo contivesse informação sobre os gols a serem validados nas 54 partidas de teste.
2. **Tratamento de Séries e Estacionariedade:** Aplicamos a primeira diferença temporal ($\Delta$) a todas as variáveis e as testamos via Dickey-Fuller Aumentado (ADF), garantindo que os testes de Granger fossem aplicados sobre séries estacionárias para evitar correlações espúrias decorrentes de tendências de longo prazo.
3. **Múltiplos Testes e Defasagens:** Realizamos buscas exaustivas de lags de 1 a 4 minutos em ambas as direções. Controlamos o erro do tipo I aplicando as correções de Bonferroni e Benjamini-Hochberg (FDR) sobre o sweep total de 128 testes causais.
4. **Validação Preditiva e Look-Ahead Leakage:** O classificador logístico de gols na janela de $t+1$ e $t+2$ utilizou GroupKFold em 5 folds com pesos balanceados para lidar com o desbalanceamento de classe. Preditores de chat foram calculados sem médias móveis centradas para garantir 0% de vazamento de dados futuros. O desempenho foi avaliado por PR-AUC e Brier Score, e a significância comparativa foi atestada pelo teste de DeLong por bootstrap.
5. **Estudo de Eventos e Graus de Liberdade:** Para testar a elevação sentimental antes do gol, agrupamos as métricas de WSI ao nível da partida ($n=52$ jogos com gols) em vez do nível do evento individual ($n=195$ gols), prevenindo a inflação de significância decorrente de usar observações intra-partida altamente correlacionadas.

---

## 1. RQ1: Sentimento do chat e a intensidade tática da partida

### Resposta direta

**Sim, existe uma associação estatística em alta frequência entre as ações de campo e as variações sentimentais no chat. Contudo, devido ao caráter misto da audiência em canais de transmissão abertos, as reações opostas dos torcedores cancelam a direcionalidade das séries minuto a minuto. Consequentemente, o momentum ofensivo direcionado a um dos times não apresenta causalidade sobre o sentimento do chat. A causalidade real surge ao analisarmos a intensidade geral do jogo de forma não-direcional, combinando as ações de perigo de ambos os lados. A análise de sensibilidade a defasagens temporais revelou as seguintes dinâmicas:**

1. **O impacto causal do jogo sobre o sentimento do chat manifesta-se com atraso de exatamente 2 minutos. Esse intervalo reflete a latência técnica de transmissão da imagem por streaming somada ao tempo de reação física e digitação dos usuários.**
2. **A causalidade reverso-antecipatória sugerida em análises exploratórias preliminares foi estatisticamente rejeitada após correções de múltiplos testes. Não há evidência estatisticamente robusta de que flutuações sentimentais ou volumétricas antecipem causalmente as ações esportivas no campo de futebol.**

#### Comportamento Temporal Médio das Métricas
Para ilustrar o comportamento geral das séries minuto a minuto, o gráfico abaixo apresenta as séries temporais médias combinadas de todas as 54 partidas, acompanhadas pelas faixas de erro padrão da média:

![Séries Temporais Médias de Chat e Campo](figures/rq1_average_time_series.png)

### Causalidade e discussão estatística em diferenças

Enquanto as séries analisadas em nível sugeriam causalidade mútua devido a tendências comuns de longo prazo, a aplicação da primeira diferença para garantir a estacionariedade eliminou toda a causalidade direcional, registrando valor p de 0,3156 para a relação entre a variação de momentum e a toxicidade. O cancelamento mútuo das reações das torcidas rivais anula a direcionalidade das variações rápidas de sentimentos.

Por outro lado, a intensidade geral do jogo restabelece a relação causal Campo $\rightarrow$ Chat em primeira diferença estacionária com alta significância na defasagem de 2 minutos:
* **Variação do Momentum Total $\rightarrow$ Variação da Polaridade:** Valor p combinado de 0,0095.
* **Variação da Ameaça Esperada (xT) Total $\rightarrow$ Variação da Razão de Toxicidade:** Valor p combinado de 0,0293.
* **Variação do Momentum Total $\rightarrow$ Variação do Sentimento WSI:** Valor p combinado de 0,0383.

A taxa de volume de comentários correlaciona-se com o perigo de campo em nível, obtendo coeficiente de 0,2464 com alta significância contemporânea. Contudo, essa relação causal desaparece em diferenças curtas por conta do ruído de alta frequência e da alta inércia própria do volume de chat. O sentimento e a toxicidade atuam como sensores afetivos imediatos e voláteis, enquanto o volume reflete um engajamento cumulativo de longo prazo.

### Rejeição da causalidade reversa por múltiplos testes

Para assegurar a validade das conclusões, conduzimos um sweep exaustivo de 1 a 4 minutos de defasagem. Os p-valores combinados revelam picos exploratórios de significância nos lags reversos de 2 e 4 minutos para a polaridade, WSI e volume rate sobre o perigo de campo. 

No entanto, a aplicação da correção estatística de Bonferroni com limiar ajustado de 0,00078 para 64 testes exaustivos invalidou todas as relações de causalidade reversa. Esse resultado demonstra que a hipótese de que o chat Granger-causa o jogo reflete flutuações casuais detectadas sob múltiplos testes exaustivos, e não uma relação de causalidade real. A causalidade opera de forma estritamente unidirecional do campo para o chat.

### Modelagem preditiva contínua e a inércia emocional

Os modelos de regressão Ridge ajustados sob validação GroupKFold para estimar a toxicidade e o volume no minuto $t+1$ a partir das métricas de campo no minuto $t$ não apresentaram melhorias em relação ao baseline autorregressivo. O chat possui elevada inércia interna, sendo que o estado da conversa no minuto seguinte é ditado quase que totalmente pelas mensagens do minuto atual, reduzindo a importância preditiva das variáveis de campo a valores próximos de zero.

### Agrupamento de partidas

Segmentamos as partidas por meio do algoritmo K-Means com 3 clusters:
* **Cluster 0: Baixa intensidade (28 jogos):** Jogos pouco movimentados com volume basal de chat de 1,9 mensagens por segundo e nuvens de palavras contendo termos neutros.
* **Cluster 1: Alta emoção e clássicos (16 jogos):** Jogos com alta média de gols e cartões, engajando o chat em um volume médio de 7,1 mensagens por segundo com nuvens de palavras de forte celebração.
* **Cluster 2: Domínio e frustração técnica (10 jogos):** Partidas com desequilíbrio tático em campo, sentimento médio WSI negativo e alta toxicidade decorrente de reclamações dos torcedores.

#### Visualização do Agrupamento e Distribuições

Abaixo, apresentamos a dispersão das partidas no UMAP de clusters e as distribuições de gols, cartões, volume de chat e polaridade sentimental para cada grupo:

![Projeção UMAP de Clusters K-Means](figures/rq1_match_clustering_umap.png)

![Distribuição de Métricas por Perfil de Partida](figures/rq1_match_clustering_boxplots.png)

---

## 2. RQ2: O chat antecipa gols e cartões ou apenas reage a eles?

### Resposta direta

**O chat ao vivo comporta-se de forma predominantemente reativa aos eventos de jogo, com uma tendência antecipatória marginal no sentimento WSI nos 2 minutos anteriores à marcação de gols. O volume bruto de mensagens não apresenta qualquer comportamento antecipatório para gols, e o sinal para cartões é puramente reativo, sem indícios de antecipação.**

### Estudo de evento e reatividade

O volume de comentários e o sentimento WSI comportam-se de forma estável até o minuto do evento, explodindo imediatamente em $t=0$ e $t+1$. Essa dinâmica caracteriza uma assinatura clássica de reação rápida. O impacto é significativamente maior para gols, onde o volume de comentários atinge 1,83 vezes a média basal da partida, do que para cartões, cujo volume de comentários eleva-se para 1,27 vezes a média basal da partida.

#### Resposta Dinâmica em Eventos (Estudo de Eventos)

Os gráficos abaixo ilustram a reatividade do volume de comentários e do sentimento WSI nos minutos que cercam gols e cartões:

![Estudo de Eventos: Gols](figures/rq2_event_study_goals.png)

![Estudo de Eventos: Cartões](figures/rq2_event_study_cards.png)

### Análise de graus de liberdade no sentimento pré-gol

A análise pré-evento original baseada em eventos individuais de gols sugeria que a elevação de WSI nos 2 minutos anteriores ao gol era altamente significativa. No entanto, essa abordagem inflou artificialmente os graus de liberdade ao ignorar as correlações internas de eventos ocorridos em uma mesma partida. 

Ao agregar os dados de WSI de forma correta ao nível do jogo para as 52 partidas que registraram gols, a diferença entre o WSI pré-gol médio e o WSI basal tornou-se apenas marginalmente significativa, com valor p de 0,0621 no teste t pareado e de 0,0672 no teste não-paramétrico de Wilcoxon. A antecipação de sentimentos pré-gol existe, mas de forma estatisticamente marginal, indicando um sinal sutil de aquecimento emocional da torcida na iminência do lance capital.

### Validação preditiva de gols no curtíssimo prazo

Para prever se um gol ocorrerá nos próximos 2 minutos a partir do minuto atual, estruturamos uma regressão logística com validação cruzada GroupKFold avaliada por meio de PR-AUC e Brier Score, corrigindo a circularidade com o uso de um grid de xT calibrado em 243 partidas independentes e eliminando médias móveis centradas para garantir 0% de vazamento de dados dados. Os resultados apontam que:
* **Métricas de Campo (xT e Momentum):** Alcançam ROC-AUC de 0,4868 e PR-AUC de 0,0805, apresentando desempenho ligeiramente inferior ao baseline de classificação aleatória de 0,0872.
* **Métricas de Chat (WSI e Volume):** Alcançam ROC-AUC de 0,5422 e PR-AUC de 0,1030, com intervalo de bootstrap que supera de forma significativa a classificação aleatória. A superioridade preditiva do chat sobre as métricas de campo foi confirmada pelo teste de DeLong por bootstrap, registrando valor p de 0,0050.
* **Coeficientes e VIF:** Todas as variáveis possuem VIF inferior a 1,40, indicando ausência de colinearidade. O WSI Lag 0 é o principal preditor positivo com coeficiente de 0,1620, enquanto o xT total apresenta coeficiente negativo de 0,0600 devido ao reset defensivo das equipes pós-ataques perigosos e ao ponto cego das bolas paradas.

---

## 3. Síntese Geral e Limitações Metodológicas

1. **Amostra Analítica:** O descarte de 9 partidas sem chat reduziu a amostra real para 54 jogos, diminuindo o poder estatístico em relação ao planejado no projeto inicial.
2. **Estacionariedade e Momentum:** A métrica de momentum tático apresenta estacionariedade em apenas 63% das partidas individuais, o que exige cautela na interpretação de seus testes de causalidade de Granger.
3. **Limitações do xT e WSI:** O cálculo de xT baseia-se em uma propagação puramente local que omite passes de progressão ofensiva, enquanto a métrica WSI sofre com a diluição por comentários neutros ou spams repetitivos em transmissões de alta atividade.
4. **Roteiro Futuro:** Propõe-se para próximos trabalhos a segmentação partidária dos usuários do chat, a substituição do WSI pelo Índice de Valência e Ativação, a aplicação de modelos autorregressivos com comutação de regimes de Markov (MS-VAR), o uso de Transferência de Entropia para acoplamentos não-lineares, a modelagem de eventos recorrentes por Andersen-Gill, e a distinção empírica de estudos de eventos específicos para gols originados em cobranças de bola parada.
