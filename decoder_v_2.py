import librosa  # для загрузки аудиофайлов
import numpy as np  # для работы с массивами
import onnxruntime as rt  # для запуска ONNX-моделей
import scipy.signal
from pathlib import Path
from typing import List, Tuple
import re 

class OnnxConformerRNNT:
    def __init__(self, model_files: dict):
        # Инициализация ONNX-сессий для энкодера и декодера
        self._encoder = rt.InferenceSession(model_files["encoder"], providers=["CPUExecutionProvider"])
        self._decoder_joint = rt.InferenceSession(model_files["decoder_joint"], providers=["CPUExecutionProvider"])
        
        # Загрузка словаря (вокабуляра)
        self.vocab = self._load_vocab(model_files["vocab"])
        
        # Настройка индексов специальных токенов
        self._setup_token_indices()
        
        # Вывод информации о модели
        self._print_model_info()

    def _load_vocab(self, vocab_path: str) -> List[str]:
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
        print(f"First 10 tokens: {self.vocab[:10]}")
        print(f"Last 10 tokens: {self.vocab[-10:]}")
        print(f"Blank index: {self._blank_idx} ('{self.vocab[self._blank_idx]}')")
        print(f"UNK index: {self._unk_idx} ('{self.vocab[self._unk_idx]}')")
        print(f"Max vocab index: {self._max_vocab_idx}")

    def _encode(self, features: np.ndarray, features_lens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Запуск энкодера ONNX и получение выходных признаков
        encoder_out, encoder_out_lens = self._encoder.run(
            ["outputs", "encoded_lengths"],
            {"audio_signal": features, "length": features_lens}
        )
        return encoder_out, encoder_out_lens

    def _decode(self, prev_tokens: List[int], prev_state: Tuple[np.ndarray, np.ndarray], encoder_out: np.ndarray) -> Tuple[np.ndarray, int, Tuple[np.ndarray, np.ndarray]]:
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
        return np.squeeze(outputs), -1, (state1, state2)

    def greedy_search(self, encoder_out: np.ndarray, encoder_out_len: np.ndarray) -> str:
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

        return self._postprocess(hyp)

    def _postprocess(self, decoded_ids: List[int]) -> str:
        # Удаление специальных токенов и преобразование индексов в текст
        valid_tokens = [self.vocab[tok_id] for tok_id in decoded_ids if tok_id not in self._tokens_to_filter and tok_id <= self._max_vocab_idx]
        text = "".join(valid_tokens).replace("▁", " ").strip()

        # Очистка текста от лишних пробелов и повторов символов
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)

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
        text = re.sub(r'\s+', ' ', text)  # Убираем лишние пробелы
        text = re.sub(r'([а-яА-Я])\1{2,}', r'\1\1', text)  # Убираем повторения букв более 2 раз
        text = re.sub(r'([.,!?])\1+', r'\1', text)  # Убираем повторения знаков препинания
        
        return text