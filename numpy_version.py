from preprocessors import LogMelSpectrogram
from decoder_v_2 import OnnxConformerRNNT
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt


# Загрузка аудиофайла
audio_path = "/home/artem/Музыка/stix.wav"
waveform, sample_rate = librosa.load(audio_path, sr=16000)

audio = waveform.reshape(1, -1)  # добавляем размерность батча
audio_lens = np.array([waveform.shape[0]], dtype=np.int64)  # длина сигнала
# Создаем экземпляр препроцессора
preprocessor = LogMelSpectrogram(n_mels=80)
# Преобразуем в мел-спектрограмму
log_mel_spectrogram, lengths = preprocessor.get_features(
    waveforms=waveform.reshape(1, -1),  # добавляем размерность батча
    waveforms_lens=np.array([len(waveform)])  # длина сигнала
)


# Извлечение признаков
features, features_len = preprocessor.get_features(audio, audio_lens)
# Преобразование в float32 для ONNX
features_np = features.astype(np.float32)
features_len_np = features_len.astype(np.int64)

# --- 3. Инициализация модели и выполнение инференса ---
model_files = {
    "encoder": "nemo-onnx/encoder-model.onnx",  # путь к ONNX-энкодеру
    "decoder_joint": "nemo-onnx/decoder_joint-model.onnx",  # путь к ONNX-декодеру
    "vocab": "nemo-onnx/vocab.txt"  # путь к файлу со словарём
}
# Создание экземпляра модели
onnx_model = OnnxConformerRNNT(model_files)

# Получение выходов энкодера
encoder_out, encoder_out_len = onnx_model._encode(features_np, features_len_np)
print(f"Encoder output shape: {encoder_out.shape}")
print(f"Encoder output length: {encoder_out_len}")

# --- 6. Декодирование аудио в текст ---
text = onnx_model.greedy_search(encoder_out, encoder_out_len)
print(text)
# print(lengths)