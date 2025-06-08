from preproccesor_decode import rnnt_nemo
import librosa
import numpy as np

# Загрузка аудиофайла
audio_path = "/home/artem/Музыка/telegram.wav"
waveform, sample_rate = librosa.load(audio_path, sr=16000)

audio = waveform.reshape(1, -1)  # добавляем размерность батча
audio_len = np.array([len(waveform)])

# Инициализация модели с путями к файлам
model_files = {
    "encoder": "nemo-onnx/encoder-model.onnx",
    "decoder_joint": "nemo-onnx/decoder_joint-model.onnx",
    "vocab": "nemo-onnx/vocab.txt"
}

# Создаем экземпляр класса rnnt_nemo с путями к моделям
processor = rnnt_nemo(model_files)

# Запускаем полный пайплайн распознавания
text = processor.transcribe(audio, audio_len)

print("Распознанный текст:", text)
