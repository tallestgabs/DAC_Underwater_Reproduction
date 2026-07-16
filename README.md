# DAC Underwater Reproduction

Reprodução e análise crítica do método DAC (Differential Attenuation Compensation), proposto por Liu et al. (2022) em *"Underwater Image Enhancement Based on Color Different Attenuation"*, aplicado ao dataset público UIEB.

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos

Certifique-se de ter o **Python 3** e o gerenciador de pacotes **pip** instalados no seu sistema.

### 2. Crie um ambiente virtual (recomendado)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
```

### 3. Instale as dependências

Todas as dependências do projeto estão listadas em `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Baixe o dataset UIEB

O dataset **não está incluído neste repositório** (por tamanho — cada pasta tem ~700MB) e precisa ser baixado separadamente:

- Link: [UIEB Dataset](https://drive.google.com/drive/folders/1soYDTu4VpqveC08346BiZ9UCyAx-zlSQ?usp=sharing)
- Caso o link acima não funcione, o dataset UIEB é público e pode ser encontrado facilmente com uma busca por "UIEB underwater image enhancement benchmark dataset".

### 5. Estrutura de Diretórios

Depois de baixar o dataset, organize as pastas na raiz do projeto conforme abaixo. **As pastas `UIEB_Raw/` e `UIEB_Reference/` não vêm no repositório** (estão no `.gitignore`) — você precisa criá-las e preencher manualmente com as imagens baixadas:

```text
DAC_Underwater_Reproduction/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── UIEB_Raw/          # Insira aqui as 890 imagens subaquáticas originais
└── UIEB_Reference/    # Insira aqui as 890 imagens de referência correspondentes (mesmo nome de arquivo do par em UIEB_Raw)
```

### 6. Rode o código

```bash
python3 main.py
```

## Resultados

Ao final da execução, o script gera:

- **Terminal**: resumo com as médias finais de MSE, PSNR, SSIM e UCIQE
- **`/output`**: as 890 imagens processadas pelo método DAC (gerado automaticamente, não versionado)
- **`metrics_results.csv`**: métricas detalhadas por imagem (MSE, PSNR, SSIM, UCIQE)

## Métricas Calculadas

| Métrica | Referência necessária? | O que mede |
|---|---|---|
| MSE | Sim | Erro quadrático médio pixel a pixel |
| PSNR | Sim | Relação sinal-ruído derivada do MSE |
| SSIM | Sim | Similaridade estrutural |
| UCIQE | Não | Qualidade perceptual (contraste, saturação, crominância) |

## Estrutura do Projeto

```text
DAC_Underwater_Reproduction/
├── main.py               # Script principal: processamento + cálculo de métricas
├── requirements.txt       # Dependências do projeto
├── .gitignore
├── README.md
├── UIEB_Raw/              # Dataset (não versionado, ver Passo 4)
├── UIEB_Reference/        # Dataset (não versionado, ver Passo 4)
├── output/                # Imagens processadas (gerado automaticamente)
└── metrics_results.csv    # Resultado detalhado (gerado automaticamente)
```

## ⚠️ Observações

- Este projeto é uma reprodução acadêmica do método descrito no artigo original. O artigo não disponibiliza código-fonte nem especifica todos os parâmetros numéricos utilizados (ex: *clip limit* do CLAHE, parâmetros do filtro bilateral) — os valores usados aqui foram calibrados empiricamente e estão documentados no relatório do projeto.
- Os resultados quantitativos obtidos nesta reprodução divergem dos reportados no artigo original; essa discrepância e as hipóteses investigadas estão detalhadas no relatório técnico do projeto.
