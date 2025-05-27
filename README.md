# 🗣️ Arabic Neural TTS Demo

This project is a **Text-to-Speech (TTS)** demo for **Modern Standard Arabic**, powered by the pretrained model [`MBZUAI/speecht5_tts_clartts_ar`](https://huggingface.co/MBZUAI/speecht5_tts_clartts_ar) using 🤗 Transformers and Streamlit.

🎯 **Goal**: Convert Arabic text into speech with the option to generate it in different speaker voices using x-vector embeddings.

---

## 🚀 Features

- 🔊 Text-to-Speech synthesis for Arabic
- 🎙️ Supports random speaker voice generation via x-vector embeddings
- 📄 User-friendly Streamlit interface
- 📥 Audio playback & download
- 🌟 Voice quality rating system
- 📚 Educational info on SpeechT5 and x-vectors

---

## 🖥️ Demo Screenshot

![demo screenshot](C:\Users\Tsieb\Desktop\M1 AI\s2\ML&NN\mini project 2\sp_tp\Capture d’écran 2025-05-27 200257.png)

---

## 📦 Requirements

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## The main packages used:

- transformers
- datasets
- torch
- soundfile
- streamlit
- numpy
- librosa

## ▶️ How to Run the App

- 1.Clone the repository:

```bash
git clone https://github.com/mohamed-taib/Arabic-Neural-TTS-Demo.git
cd Arabic-Neural-TTS-Demo
```

- 2.Install the dependencies:

```bash
pip install -r requirements.txt
```

- 3.Launch the Streamlit app:

```bash
streamlit run app.py
```

## 🌍 Sample Sentences

You can try the app with the following built-in examples:

مرحبًا بكم في هذا المشروع الصوتي

الذكاء الاصطناعي يغير العالم من حولنا

تعلم اللغة العربية يمنحك فرصة جديدة

الصوت البشري يحمل الكثير من المشاعر

البيانات الصوتية ضرورية لتطوير الأنظمة الذكية

## 👨‍💻 Developer Info

Name: BENAICHA Mohamed Etaib
📊 Kaggle Profile
🔗 LinkedIn

## 📄 License

This project is licensed for educational purposes and uses open models from Hugging Face.
