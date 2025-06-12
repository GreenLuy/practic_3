# Импорт необходимых библиотек
import librosa  # для загрузки аудиофайлов
import numpy as np  # для работы с массивами
import onnxruntime as rt  # для запуска ONNX-моделей
import scipy.signal
from pathlib import Path
from typing import List, Tuple
import re  # регулярные выражения для постобработки текста

# Класс препроцессора на NumPy (вместо NeMo)
class Preprocessor:
    def __init__(self, n_mels: int = 80):
        # Параметры предобработки (как в setup.py)
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.preemph = 0.97
        self.n_mels = n_mels
        self.log_zero_guard_value = float(2**-24)
        
        # Создание мел-фильтров
        self.melscale_fbanks = self.create_mel_filterbank()
        
        # Создание окна Ханна
        self.hann_window = self.create_hann_window()
    
    def create_mel_filterbank(self) -> np.ndarray:
        """Создание мел-фильтров"""
        n_freqs = self.n_fft // 2 + 1
        
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        
        def mel_to_hz(mel):
            return 700.0 * (10.0**(mel / 2595.0) - 1.0)
        
        # Создание точек в мел-шкале
        low_freq_mel = hz_to_mel(0)
        high_freq_mel = hz_to_mel(self.sample_rate // 2)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Преобразование в индексы частотных бинов
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Создание треугольных фильтров
        fbank = np.zeros((self.n_mels, n_freqs))
        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            
            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus:
                    fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m:
                    fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])
        
        return fbank.T
    
    def create_hann_window(self) -> np.ndarray:
        """Создание окна Ханна с паддингом"""
        hann = scipy.signal.windows.hann(self.win_length, sym=False)
        pad_size = self.n_fft // 2 - self.win_length // 2
        padded_hann = np.pad(hann, (pad_size, pad_size), mode='constant', constant_values=0)
        return padded_hann
    
    def preemphasis(self, waveforms: np.ndarray) -> np.ndarray:
        """Применение предыскажения"""
        if self.preemph == 0.0:
            return waveforms
        
        preemphasized = np.zeros_like(waveforms)
        preemphasized[:, 0] = waveforms[:, 0]
        preemphasized[:, 1:] = waveforms[:, 1:] - self.preemph * waveforms[:, :-1]
        return preemphasized
    
    def pad_waveforms(self, waveforms: np.ndarray) -> np.ndarray:
        """Паддинг сигналов для STFT"""
        pad_size = self.n_fft // 2
        padded = np.pad(waveforms, ((0, 0), (pad_size, pad_size)), mode='reflect')
        return padded
    
    def stft(self, waveforms: np.ndarray) -> np.ndarray:
        """Вычисление STFT"""
        batch_size = waveforms.shape[0]
        spectrograms = []
        
        for b in range(batch_size):
            f, t, stft_result = scipy.signal.stft(
                waveforms[b], 
                fs=self.sample_rate,
                window=self.hann_window,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
                nfft=self.n_fft,
                return_onesided=True
            )
            
            magnitude = np.abs(stft_result) ** 2
            spectrograms.append(magnitude)
        
        return np.stack(spectrograms, axis=0)
    
    def compute_mel_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Вычисление мел-спектрограммы"""
        mel_spectrogram = np.matmul(spectrogram.transpose(0, 2, 1), self.melscale_fbanks)
        return mel_spectrogram.transpose(0, 2, 1)
    
    def log_mel_spectrogram(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Применение логарифма к мел-спектрограмме"""
        return np.log(mel_spectrogram + self.log_zero_guard_value)
    
    def normalize(self, features: np.ndarray, features_lens: np.ndarray) -> np.ndarray:
        """Нормализация признаков по времени"""
        batch_size, n_mels, max_frames = features.shape
        normalized_features = np.zeros_like(features)
        
        for b in range(batch_size):
            seq_len = int(features_lens[b])
            if seq_len > 0:
                sequence = features[b, :, :seq_len]
                mean = np.mean(sequence, axis=1, keepdims=True)
                var = np.var(sequence, axis=1, keepdims=True, ddof=1)
                normalized_sequence = (sequence - mean) / (np.sqrt(var) + 1e-5)
                normalized_features[b, :, :seq_len] = normalized_sequence
        
        return normalized_features
    
    def get_features(self, waveforms: np.ndarray, waveforms_lens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Основная функция предобработки"""
        # 1. Предыскажение
        waveforms = self.preemphasis(waveforms)
        
        # 2. Паддинг
        waveforms = self.pad_waveforms(waveforms)
        
        # 3. STFT
        spectrogram = self.stft(waveforms)
        
        # 4. Мел-спектрограмма
        mel_spectrogram = self.compute_mel_spectrogram(spectrogram)
        
        # 5. Логарифм
        log_mel_spectrogram = self.log_mel_spectrogram(mel_spectrogram)
        
        # 6. Вычисление длин признаков
        features_lens = waveforms_lens // self.hop_length + 1
        
        # 7. Нормализация
        normalized_features = self.normalize(log_mel_spectrogram, features_lens)
        
        return normalized_features, features_lens

class OnnxConformerRNNT:
    def __init__(self, model_files: dict):
        # Инициализация ONNX-сессий для энкодера и декодера
        self._encoder = rt.InferenceSession(model_files["encoder"], providers=["CPUExecutionProvider"])
        self._decoder_joint = rt.InferenceSession(model_files["decoder_joint"], providers=["CPUExecutionProvider"])
        
        # Загрузка словаря
        self.vocab = self.load_vocab(model_files["vocab"])
        
        # Настройка индексов специальных токенов
        self._setup_token_indices()
        
        # Вывод информации о модели
        self.print_model_info()

    def load_vocab(self, vocab_path: str) -> List[str]:
        # Загрузка словаря из файла
        vocab = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                # Поддержка как формата "токен индекс", так и "токен"
                if len(parts) == 2:
                    token, idx = parts
                    idx = int(idx)
                    while len(vocab) <= idx:
                        vocab.append("")
                    vocab[idx] = token
                elif len(parts) == 1:
                    vocab.append(parts[0])
        return vocab

    def _setup_token_indices(self):
        # Индексы токенов
        self._blank_idx = self.vocab.index("<blk>") if "<blk>" in self.vocab else 1024
        self._max_vocab_idx = len(self.vocab) - 1
        # Токены, которые нужно фильтровать после декодирования
        self._tokens_to_filter = {self._blank_idx}
        for i in range(self._max_vocab_idx + 1, len(self.vocab)):
            self._tokens_to_filter.add(i)

    def print_model_info(self):
        # Печать информации о загруженном словаре
        print(f"Loaded vocabulary with {len(self.vocab)} tokens")
        print(f"First 10 tokens: {self.vocab[:10]}")
        print(f"Last 10 tokens: {self.vocab[-10:]}")
        print(f"Blank index: {self._blank_idx} ('{self.vocab[self._blank_idx]}')")
        print(f"Max vocab index: {self._max_vocab_idx}")

    def _encode(self, features: np.ndarray, features_lens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Запуск энкодера ONNX и получение выходных признаков
        encoder_out, encoder_out_lens = self._encoder.run(
            ["outputs", "encoded_lengths"],
            {"audio_signal": features, "length": features_lens}
        )
        return encoder_out, encoder_out_lens

    def decode(self, prev_tokens: List[int], prev_state: Tuple[np.ndarray, np.ndarray], encoder_out: np.ndarray) -> Tuple[np.ndarray, int, Tuple[np.ndarray, np.ndarray]]:
        # Запуск декодера (joint network) с текущими входами и состояниями
        outputs, state1, state2 = self._decoder_joint.run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": encoder_out.astype(np.float32),
                "targets": np.array([[self._blank_idx if not prev_tokens else prev_tokens[-1]]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "input_states_1": prev_state[0],
                "input_states_2": prev_state[1],
            }
        )
        return np.squeeze(outputs), -1, (state1, state2)

    def greedy_search(self, encoder_out: np.ndarray, encoder_out_len: np.ndarray) -> str:
        # Простая стратегия декодирования (поиск наиболее вероятных токенов)
        max_len = encoder_out.shape[2]
        print(f"Starting greedy decoding with {max_len} time frames")

        # Начальное состояние RNN
        state = (np.zeros((1, 1, 640), dtype=np.float32), np.zeros((1, 1, 640), dtype=np.float32))
        hyp = []  # гипотеза: список выбранных токенов
        
        for t in range(max_len):
            current_encoder_out = encoder_out[:, :, t:t + 1]  # текущий временной шаг
            logits, _, state = self.decode(hyp, state, current_encoder_out)
            logits = logits.copy()
            
            # Обрезаем массив логарифмов до 1025 элементов
            logits = logits[:1025]
            
            next_token = np.argmax(logits).item()            
            # Добавление токена, если он не пустой и не повтор предыдущего
            if next_token != self._blank_idx and (not hyp or next_token != hyp[-1]):
                hyp.append(next_token)

        return self.postprocess(hyp)

    def postprocess(self, decoded_ids: List[int]) -> str:
        # Удаление специальных токенов и преобразование индексов в текст
        valid_tokens = [self.vocab[tok_id] for tok_id in decoded_ids if tok_id not in self._tokens_to_filter and tok_id <= self._max_vocab_idx]
        text = "".join(valid_tokens).replace("▁", " ").strip()

        # Очистка текста от лишних пробелов и повторов символов
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)

        # Удаление повторяющихся слов, если они длинные
        words = text.split()
        cleaned_words = []
        last_word = None
        for word in words:
            if word and (word.lower() != (last_word or "").lower() or len(word) <= 2):
                cleaned_words.append(word)
                last_word = word
        text = " ".join(cleaned_words).strip()
        return text

# Загрузка аудио с частотой дискретизации 16000 Гц
audio, sr = librosa.load("telegram.wav", sr=16000)

# Подготовка данных для препроцессора
audio_batch = audio.reshape(1, -1)
audio_len = np.array([audio.shape[0]], dtype=np.int64)

# Создание препроцессора
preprocessor = Preprocessor(n_mels=80)

# Извлечение признаков
features, features_len = preprocessor.get_features(audio_batch, audio_len)
print(f"Features shape: {features.shape}")
print(f"Features length: {features_len}")

# Преобразование в float32 для ONNX
features_np = features.astype(np.float32)
features_len_np = features_len.astype(np.int64)

# Загрузка моделей ONNX
model_files = {
    "encoder": "encoder-model.onnx",  # путь к ONNX-энкодеру
    "decoder_joint": "decoder_joint-model.onnx",  # путь к ONNX-декодеру
    "vocab": "vocab.txt"  # путь к файлу со словарём
}

# Создание экземпляра модели
onnx_model = OnnxConformerRNNT(model_files)

# Получение выходов энкодера
encoder_out, encoder_out_len = onnx_model._encode(features_np, features_len_np)
print(f"Encoder output shape: {encoder_out.shape}")
print(f"Encoder output length: {encoder_out_len}")

# --- 6. Декодирование аудио в текст ---
text = onnx_model.greedy_search(encoder_out, encoder_out_len)
print("Transcription:", text)