# Импорт необходимых библиотек
import librosa  # для загрузки аудиофайлов
import numpy as np  # для работы с массивами
import onnxruntime as rt  # для запуска ONNX-моделей
import scipy.signal
from pathlib import Path
from typing import List, Tuple
import re  # регулярные выражения для постобработки текста

# Класс препроцессора на NumPy (вместо NeMo)
class LogMelSpectrogram:
    def __init__(self, n_mels: int = 80):
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.preemph = 0.97
        self.n_mels = n_mels
        self.log_zero_guard_value = float(2**-24)
        
        # Создание мел-фильтров
        self.melscale_fbanks = self._create_mel_filterbank()
        
        # Создание окна Ханна
        self.hann_window = self._create_hann_window()
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """Создание мел-фильтров"""
        n_freqs = self.n_fft // 2 + 1
        
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        
        def mel_to_hz(mel):
            return 700.0 * (10.0**(mel / 2595.0) - 1.0)
        
        # Создание точек в мел-шкале
        low_freq_mel = hz_to_mel(0)
        high_freq_mel = hz_to_mel(self.sample_rate // 2)
        mel_points = np.linspace(int(low_freq_mel), int(high_freq_mel), self.n_mels + 2)
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
    
    def _create_hann_window(self) -> np.ndarray:
        """Создание окна Ханна с паддингом"""
        hann = scipy.signal.windows.hann(self.win_length, sym=False)
        pad_size = self.n_fft // 2 - self.win_length // 2
        padded_hann = np.pad(hann, (pad_size, pad_size), mode='constant', constant_values=0)
        return padded_hann
    
    def _preemphasis(self, waveforms: np.ndarray) -> np.ndarray:
        """Применение предыскажения"""
        if self.preemph == 0.0:
            return waveforms
        
        preemphasized = np.zeros_like(waveforms)
        preemphasized[:, 0] = waveforms[:, 0]
        preemphasized[:, 1:] = waveforms[:, 1:] - self.preemph * waveforms[:, :-1]
        return preemphasized
    
    def _pad_waveforms(self, waveforms: np.ndarray) -> np.ndarray:
        """Паддинг сигналов для STFT"""
        pad_size = self.n_fft // 2
        padded = np.pad(waveforms, ((0, 0), (pad_size, pad_size)), mode='reflect')
        return padded
    
    def _stft(self, waveforms: np.ndarray) -> np.ndarray:
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
    
    def _compute_mel_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Вычисление мел-спектрограммы"""
        mel_spectrogram = np.matmul(spectrogram.transpose(0, 2, 1), self.melscale_fbanks)
        return mel_spectrogram.transpose(0, 2, 1)
    
    def _log_mel_spectrogram(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Применение логарифма к мел-спектрограмме"""
        return np.log(mel_spectrogram + self.log_zero_guard_value)
    
    def _normalize(self, features: np.ndarray, features_lens: np.ndarray) -> np.ndarray:
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
        waveforms = self._preemphasis(waveforms)
        
        # 2. Паддинг
        waveforms = self._pad_waveforms(waveforms)
        
        # 3. STFT
        spectrogram = self._stft(waveforms)
        
        # 4. Мел-спектрограмма
        mel_spectrogram = self._compute_mel_spectrogram(spectrogram)
        
        # 5. Логарифм
        log_mel_spectrogram = self._log_mel_spectrogram(mel_spectrogram)
        
        # 6. Вычисление длин признаков
        features_lens = waveforms_lens // self.hop_length + 1
        
        # 7. Нормализация
        normalized_features = self._normalize(log_mel_spectrogram, features_lens)
        
        return normalized_features, features_lens