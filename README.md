# LLM 

## 環境構築

1. Python 環境構築
1. Python 仮想環境設定
1. Python ライブラリインストール
1. Python スクリプト実行

### 1. Python 環境構築

```shell
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential
```

### Python 仮想環境設定

```shell
python3 -m venv venv

# 仮想環境を有効化（アクティベート）
source venv/bin/activate
```

### Python ライブラリインストール

```shell
pip install --upgrade pip

# CPU版のPyTorch一式をインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Python スクリプト実行

```shell
python3 sample.py
```

LLM の学習用


