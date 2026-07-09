# DAC Underwater Reproduction

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos

Certifique-se de ter o **Python 3** e o gerenciador de pacotes **pip** instalados no seu sistema.

### Crie um ambiente virtual
```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Rode o comando abaixo para instalar as dependências:
```bash
pip install opencv-python numpy scikit-image
```

### Baixe o UIEB Dataset clicando [AQUI](https://drive.google.com/drive/folders/1soYDTu4VpqveC08346BiZ9UCyAx-zlSQ?usp=sharing)


### 2. Estrutura de Diretórios

O script espera que os datasets estejam organizados na raiz do projeto conforme a estrutura abaixo. As pastas de imagens brutas e de referência devem ser baixadas previamente (Se o link não estiver funcionando, o dataset UIEB é público).

```text
trabalhoFinal/
├── main.py
├── UIEB_Raw/          # Insira aqui as 890 imagens subaquáticas originais 
└── UIEB_Reference/    # Insira aqui as 890 imagens de referência correspondentes
```

### Rode o código com o comando abaixo:
```bash
python3 main.py
```

### Resultados em:
- Terminal
- /output
- metrics_results.csv


