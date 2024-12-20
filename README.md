# LLM_Projects
This repository contains my specialized projects related to LLM training, parallel GPU training, Retrieval-Augmented Generation (RAG) systems, and intelligent agents.


-----------
# GitHub'a Proje Yükleme Adımları

Bu rehber, Linux'ta Visual Studio Code ve Git kullanarak projelerinizi GitHub'a nasıl yükleyeceğinizi adım adım açıklar.

---

## 1. Git Kurulumu
Öncelikle Git'in kurulu olduğundan emin olun:

```bash
sudo apt update
sudo apt install git
git --version

git config --global user.name "Adınız"
git config --global user.email "E-posta adresiniz"


ssh-keygen -t ed25519 -C "e-posta@example.com"

cat ~/.ssh/id_ed25519.pub


GitHub'a SSH Anahtarı Ekle:
GitHub'da oturum açın.
Settings > SSH and GPG keys > New SSH key yolunu izleyin.
Public anahtarı kopyalayıp yapıştırın ve Add SSH key'e tıklayın.

cd /proje/klasörü/yolu
git init
git remote add origin https://github.com/kullaniciadi/depoadi.git
git add .
git commit -m "İlk commit"
git branch -M main
git push -u origin main
git add .
git commit -m "Commit mesajı"
git push
ssh -T git@github.com
