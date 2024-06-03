echo "Xin chao anh em!"
echo "Chung toi dang setup moi truong ..."
echo "..."
pip install -q --upgrade pip
pip install -q -U datasets
pip install -q transformers trl
pip install -q -r "/kaggle/working/dialogue-text-summarization/requirements.txt"
pip install -q -r "/content/dialogue-text-summarization/requirements.txt"
pip install -q -r "requirements.txt"
echo "---------"
echo "Set up complete!"