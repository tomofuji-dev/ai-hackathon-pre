# 0. セットアップ

## 0.0. 前提

- MacOS でのセットアップを想定

## 0.1. リポジトリのクローン

```
cd ~/Desktop
git clone https://github.com/tomofuji-dev/ai-hackathon-pre.git
cd ./ai-hackathon-pre
```

## 0.2. asdf のインストール

```
brew install asdf
echo 'export PATH="${ASDF_DATA_DIR:-$HOME/.asdf}/shims:$PATH"' >> ~/.zshrc
```

## 0.3. asdf で Python, poetry のインストール

```
asdf plugin-add python
asdf plugin-add poetry
asdf install
```

## 0.4. poetry のパッケージインストール

```
poetry install
```

# 1. `./lab`

```
poetry run jupyter lab
```

# 2. `./backend`

```
poetry run uvicorn main:app --reload
```
