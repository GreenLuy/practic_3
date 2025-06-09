import numpy as np
import onnxruntime as rt  # для запуска ONNX-моделей

class RNNTDecoder:
    def __init__(self, model_files):
        # Инициализация ONNX-сессий для энкодера и декодера
        self._encoder = rt.InferenceSession(model_files["encoder"], providers=["CPUExecutionProvider"])
        self._decoder_joint = rt.InferenceSession(model_files["decoder_joint"], providers=["CPUExecutionProvider"])
        
        # Загрузка словаря
        self.vocab = self._load_vocab(model_files["vocab"])
        
        # Настройка индексов специальных токенов
        self._setup_token_indices()
        
        # Вывод информации о модели
        self._print_model_info()

    def _load_vocab(self, vocab_path):
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
        # Дополнение словаря, если в нём меньше 2561 элементов
        while len(vocab) < 2561:
            vocab.append("<pad>")
        return vocab

    def _setup_token_indices(self):
        # Индексы специальных токенов
        self._blank_idx = self.vocab.index("<blk>") if "<blk>" in self.vocab else 2560
        self._unk_idx = self.vocab.index("<unk>") if "<unk>" in self.vocab else 0
        self._blk_idx = self._blank_idx
        self._pad_idx = self.vocab.index("<pad>") if "<pad>" in self.vocab else 2560
        self._max_vocab_idx = len(self.vocab) - 1
        # Токены, которые нужно фильтровать после декодирования
        self._tokens_to_filter = {self._blank_idx, self._unk_idx, self._blk_idx, self._pad_idx}
        for i in range(self._max_vocab_idx + 1, len(self.vocab)):
            self._tokens_to_filter.add(i)

    def _print_model_info(self):
        # Печать информации о загруженном словаре
        print(f"Loaded vocabulary with {len(self.vocab)} tokens")
        # print(f"First 10 tokens: {self.vocab[:10]}")
        # print(f"Last 10 tokens: {self.vocab[-10:]}")
        print(f"Blank index: {self._blank_idx} ('{self.vocab[self._blank_idx]}')")
        print(f"UNK index: {self._unk_idx} ('{self.vocab[self._unk_idx]}')")
        print(f"Max vocab index: {self._max_vocab_idx}")

    def _encode(self, features, features_lens):
        # Запуск энкодера ONNX и получение выходных признаков
        encoder_out, encoder_out_lens = self._encoder.run(
            ["outputs", "encoded_lengths"],
            {"audio_signal": features, "length": features_lens}
        )
        return encoder_out, encoder_out_lens

    def _decode(self, prev_tokens, prev_state, encoder_out):
        # Запуск декодера (joint network) с текущими входами и состояниями
        current_token = self._blank_idx if not prev_tokens else prev_tokens[-1]
        # Убедимся, что токен находится в допустимом диапазоне
        current_token = max(0, min(current_token, 1024))  # Ограничиваем максимальным значением 1024
        
        outputs, state1, state2 = self._decoder_joint.run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": encoder_out.astype(np.float32),
                "targets": np.array([[current_token]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "input_states_1": prev_state[0],
                "input_states_2": prev_state[1],
            }
        )
        # print("Decoder output shapes:")
        # print(f"outputs shape: {outputs.shape}")
        # print(f"state1 shape: {state1.shape}")
        # print(f"state2 shape: {state2.shape}")
        # print(f"outputs dtype: {outputs.dtype}")
        # print(f"state1 dtype: {state1.dtype}")
        # print(f"state2 dtype: {state2.dtype}")
        return np.squeeze(outputs), -1, (state1, state2)

    def greedy_search(self, encoder_out, encoder_out_len):
        # Простая стратегия декодирования (поиск наиболее вероятных токенов)
        max_len = encoder_out.shape[2]
        print(f"Starting greedy decoding with {max_len} time frames")

        # Начальное состояние RNN
        state = (np.zeros((1, 1, 640), dtype=np.float32), np.zeros((1, 1, 640), dtype=np.float32))
        hyp = []  # гипотеза: список выбранных токенов
        last_token = None
        repeat_count = 0
        
        for t in range(max_len):
            current_encoder_out = encoder_out[:, :, t:t + 1]  # текущий временной шаг
            logits, _, state = self._decode(hyp, state, current_encoder_out)
            logits = logits.copy()

            # Маскируем pad и unknown токены
            logits[self._pad_idx] = float('-inf')
            logits[self._unk_idx] = float('-inf')
            
            # Маскируем индексы вне допустимого диапазона
            valid_indices = np.arange(len(logits)) <= 1024
            logits[~valid_indices] = float('-inf')
            
            # Если текущий токен повторяется, увеличиваем счетчик
            if last_token is not None and last_token == np.argmax(logits).item():
                repeat_count += 1
                # Если токен повторился более 2 раз, сильно уменьшаем его вероятность
                if repeat_count > 2:
                    logits[last_token] *= 0.1
            else:
                repeat_count = 0
            
            next_token = np.argmax(logits).item()
            last_token = next_token
            
            # Проверяем, что токен в допустимом диапазоне
            if next_token > 1024 or next_token in {self._pad_idx, self._unk_idx}:
                continue
                
            # Добавление токена, если он не blank
            if next_token != self._blank_idx:
                # Проверяем, не создает ли токен повторяющийся паттерн
                if len(hyp) >= 2 and hyp[-1] == next_token and hyp[-2] == next_token:
                    continue
                hyp.append(next_token)
        print("tensors: ", hyp)
        return self._postprocess(hyp)

    def _postprocess(self, decoded_ids):
        # Удаление специальных токенов и преобразование индексов в текст
        valid_tokens = [self.vocab[tok_id] for tok_id in decoded_ids if tok_id not in self._tokens_to_filter and tok_id <= self._max_vocab_idx]
        text = "".join(valid_tokens).replace("▁", " ").strip()

        # Очистка текста от лишних пробелов и повторов символов
        text = " ".join(text.split())  # Убираем лишние пробелы
        
        # Исправление разделения слов
        words = text.split()
        cleaned_words = []
        last_word = None
        
        for word in words:
            # Пропускаем пустые слова
            if not word:
                continue
                
            # Если текущее слово начинается с "по" и предыдущее слово короткое (1-2 буквы)
            if word.startswith('по') and last_word and len(last_word) <= 2:
                # Объединяем с предыдущим словом
                cleaned_words[-1] = last_word + word
            # Если текущее слово короткое (1-2 буквы) и предыдущее слово заканчивается на "по"
            elif len(word) <= 2 and last_word and last_word.endswith('по'):
                # Объединяем с предыдущим словом
                cleaned_words[-1] = last_word + word
            # Если текущее слово не повторяет предыдущее
            elif not last_word or word.lower() != last_word.lower():
                cleaned_words.append(word)
                last_word = word

        text = " ".join(cleaned_words).strip()
        
        # Дополнительная очистка
        text = " ".join(text.split())  # Убираем лишние пробелы
        
        return text

class rnnt_nemo():
    def __init__(self, model_files=None):
        # Базовые параметры
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.preemph = 0.97
        self.n_mels = 80
        self.log_zero_guard_value = float(2**-24)
        self.dither = 1.0e-05
        self.frame_splicing = 1

        # Логирование конфигурации
        print("\nPreprocessor configuration:")
        print(f"Sample rate: {self.sample_rate}")
        print(f"Window size: {self.win_length} samples ({self.win_length/self.sample_rate:.3f} seconds)")
        print(f"Window stride: {self.hop_length} samples ({self.hop_length/self.sample_rate:.3f} seconds)")
        print(f"N_fft: {self.n_fft}")
        print(f"N_mels: {self.n_mels}")
        print(f"Preemph: {self.preemph}")
        print(f"Dither: {self.dither}")
        print(f"Frame splicing: {self.frame_splicing}")

        # Initialize Hann window
        self.hann_window = self._create_hann_window()

        # Create and store mel filter banks
        self.mel_filters_80 = self.mel_scale_fbanks(self.n_fft, 0, self.sample_rate // 2, self.n_mels, self.sample_rate)
        self.mel_filters_128 = self.mel_scale_fbanks(self.n_fft, 0, self.sample_rate // 2, 128, self.sample_rate)

        # Initialize decoder if model files are provided
        if model_files:
            try:
                self.decoder = RNNTDecoder(model_files)
                print("\nDecoder initialized successfully")
            except Exception as e:
                print(f"\nError initializing decoder: {e}")
                raise
        else:
            self.decoder = None
            print("\nWarning: No model files provided, decoder not initialized")

    def mel_scale_fbanks(self, n_fft, f_min, f_max, n_mels, sample_rate):
        # Преобразование частоты в мел-частоту
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)

        # Генерация мел-частот
        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Преобразование в индексы
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        # Создание фильтров
        filters = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(1, n_mels + 1):
            filters[m - 1, bin_points[m - 1]:bin_points[m]] = \
                (np.arange(bin_points[m - 1], bin_points[m]) - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
            filters[m - 1, bin_points[m]:bin_points[m + 1]] = \
                (bin_points[m + 1] - np.arange(bin_points[m], bin_points[m + 1])) / (bin_points[m + 1] - bin_points[m])

        return filters

    def _normalize(self, features, features_lens):
        """Нормализация признаков по признаку (per_feature)"""
        batch_size, n_mels, max_frames = features.shape
        normalized_features = np.zeros_like(features)
        
        for b in range(batch_size):
            seq_len = int(features_lens[b])
            if seq_len > 0:
                sequence = features[b, :, :seq_len]
                # Нормализация по каждому признаку отдельно
                for m in range(n_mels):
                    mean = np.mean(sequence[m])
                    var = np.var(sequence[m], ddof=1)
                    normalized_features[b, m, :seq_len] = (sequence[m] - mean) / (np.sqrt(var) + 1e-5)
        
        return normalized_features

    def _create_hann_window(self):
        """Создание окна Ханна с паддингом"""
        # Создаем окно Ханна
        n = np.arange(self.win_length)
        hann = 0.5 * (1 - np.cos(2 * np.pi * n / (self.win_length - 1)))
        
        # Паддинг
        pad_size = self.n_fft // 2 - self.win_length // 2
        padded_hann = np.pad(hann, (pad_size, pad_size), mode='constant', constant_values=0)
        
        return padded_hann

    def _preemphasis(self, waveforms):
        """Применение предыскажения"""
        if self.preemph == 0.0:
            return waveforms
        
        preemphasized = np.zeros_like(waveforms)
        preemphasized[:, 0] = waveforms[:, 0]
        preemphasized[:, 1:] = waveforms[:, 1:] - self.preemph * waveforms[:, :-1]
        return preemphasized

    def _pad_waveforms(self, waveforms):
        """Паддинг сигналов для STFT"""
        pad_size = self.n_fft // 2
        padded = np.pad(waveforms, ((0, 0), (pad_size, pad_size)), mode='reflect')
        return padded

    def _stft(self, waveforms):
        """Вычисление STFT без использования scipy.signal.stft"""
        batch_size = waveforms.shape[0]
        n_fft = self.n_fft
        hop_length = self.hop_length
        window = self.hann_window
        spectrograms = []

        for b in range(batch_size):
            # Получаем текущий сигнал
            signal = waveforms[b]
            # Вычисляем количество окон
            num_windows = (len(signal) - n_fft) // hop_length + 1
            # Инициализируем массив для STFT
            stft_result = np.zeros((n_fft // 2 + 1, num_windows), dtype=np.complex_)

            for i in range(num_windows):
                # Определяем начальный индекс для текущего окна
                start = i * hop_length
                # Извлекаем текущее окно и применяем окно Ханна
                windowed_signal = signal[start:start + n_fft] * window
                # Применяем БПФ
                stft_result[:, i] = np.fft.rfft(windowed_signal)

            # Вычисляем величину
            magnitude = np.abs(stft_result) ** 2
            spectrograms.append(magnitude)

        return np.stack(spectrograms, axis=0)

    def _compute_mel_spectrogram(self, spectrogram):
        """Вычисление мел-спектрограммы"""
        mel_spectrogram = np.matmul(spectrogram.transpose(0, 2, 1), self.mel_filters_80.T)  # Возвращаем к mel_filters_80
        return mel_spectrogram.transpose(0, 2, 1)

    def _log_mel_spectrogram(self, mel_spectrogram):
        """Применение логарифма к мел-спектрограмме"""
        return np.log(mel_spectrogram + self.log_zero_guard_value)

    def get_features(self, waveforms, waveforms_lens):
        """Основная функция предобработки"""
        try:
            # Проверка входных данных
            if waveforms is None or waveforms_lens is None:
                raise ValueError("Input waveforms or lengths cannot be None")
            
            if not isinstance(waveforms, np.ndarray):
                raise TypeError("Waveforms must be numpy array")
            
            if waveforms.ndim != 2:
                raise ValueError(f"Expected 2D array for waveforms, got {waveforms.ndim}D")
            
            if waveforms.shape[0] != len(waveforms_lens):
                raise ValueError(f"Batch size mismatch: waveforms {waveforms.shape[0]} != lengths {len(waveforms_lens)}")

            # 1. Добавляем дизеринг
            if self.dither > 0:
                waveforms = waveforms + np.random.normal(0, self.dither, waveforms.shape)
            
            # 2. Предыскажение
            waveforms = self._preemphasis(waveforms)
            
            # 3. Паддинг
            waveforms = self._pad_waveforms(waveforms)
            
            # 4. STFT
            spectrogram = self._stft(waveforms)
            
            # 5. Мел-спектрограмма
            mel_spectrogram = self._compute_mel_spectrogram(spectrogram)
            
            # 6. Логарифм
            log_mel_spectrogram = self._log_mel_spectrogram(mel_spectrogram)
            
            # 7. Вычисление длин признаков
            features_lens = waveforms_lens // self.hop_length + 1
            
            # 8. Нормализация (теперь per_feature)
            normalized_features = self._normalize(log_mel_spectrogram, features_lens)
            
            # 9. Frame splicing (если нужно)
            if self.frame_splicing > 1:
                normalized_features = self._frame_splicing(normalized_features, features_lens)
            
            return normalized_features, features_lens

        except Exception as e:
            print(f"\nError in get_features: {e}")
            raise

    def _frame_splicing(self, features, features_lens):
        """Применение frame splicing"""
        batch_size, n_mels, max_frames = features.shape
        spliced_features = []
        spliced_lens = []
        
        for b in range(batch_size):
            seq_len = int(features_lens[b])
            if seq_len > 0:
                sequence = features[b, :, :seq_len]
                # Создаем окна размера frame_splicing
                windows = []
                for i in range(0, seq_len - self.frame_splicing + 1):
                    window = sequence[:, i:i + self.frame_splicing].flatten()
                    windows.append(window)
                spliced = np.stack(windows, axis=1)
                spliced_features.append(spliced)
                spliced_lens.append(len(windows))
        
        # Находим максимальную длину
        max_len = max(spliced_lens)
        
        # Паддинг до максимальной длины
        padded_features = np.zeros((batch_size, n_mels * self.frame_splicing, max_len))
        for b in range(batch_size):
            padded_features[b, :, :spliced_lens[b]] = spliced_features[b]
        
        return padded_features

    def transcribe(self, audio, audio_len):
        """
        Полный пайплайн от аудио до текста
        Args:
            audio: numpy array с аудио данными
            audio_len: numpy array с длинами аудио
        Returns:
            str: распознанный текст
        """
        try:
            if self.decoder is None:
                raise ValueError("Decoder not initialized. Please provide model_files in constructor.")

            # Проверка входных данных
            if audio is None or audio_len is None:
                raise ValueError("Input audio or lengths cannot be None")
            
            if not isinstance(audio, np.ndarray):
                raise TypeError("Audio must be numpy array")
            
            if audio.ndim != 2:
                raise ValueError(f"Expected 2D array for audio, got {audio.ndim}D")
            
            if audio.shape[0] != len(audio_len):
                raise ValueError(f"Batch size mismatch: audio {audio.shape[0]} != lengths {len(audio_len)}")

            # 1. Извлечение признаков
            features, features_lens = self.get_features(audio, audio_len)

            # 2. Преобразование в float32 для ONNX
            features = features.astype(np.float32)
            features_lens = features_lens.astype(np.int64)

            # 3. Получение выходов энкодера
            encoder_out, encoder_out_len = self.decoder._encode(features, features_lens)

            print("\nEncoder output shape:", encoder_out.shape)
            print("Encoder output lengths:", encoder_out_len)

            # 4. Декодирование в текст
            text = self.decoder.greedy_search(encoder_out, encoder_out_len)

            return text

        except Exception as e:
            print(f"\nError in transcribe: {e}")
            raise