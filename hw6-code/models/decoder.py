import torch
from torch import Tensor, nn

from .attention import AdditiveAttention


class RNNDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        vocab_size: int,
        feature_dim: int,
        dropout=0.5,
        rnn_cell=nn.LSTMCell,
    ):
        """Initialize RNNDecoder.

        Args:
            embed_dim (int): Embedding dimension.
            hidden_dim (int): Hidden dimension.
            vocab_size (int): Vocabulary size.
            feature_dim (int): Feature dimension.
            dropout (float): Dropout rate. Default: 0.5.
            rnn_cell (Module): RNN cell module. Default: nn.LSTMCell.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # Decoder expects these layers:
        # - embedding: input embedding
        # - lstm: lstm cell
        # - init_h: linear projection to init hidden state from image features
        # - init_c: linear projection to init cell state from image features
        # - dropout: output dropout layer
        # - fc: output linear projection
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = rnn_cell(embed_dim, hidden_dim)
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Initialize hidden states for RNNDecoder.

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).

        Returns:
            tuple: (hidden_state, cell_state) both with shape (batch, hidden_dim).
        """
        mean_features = features.mean(dim=1)
        h = self.init_h(mean_features)
        c = self.init_c(mean_features)
        return h, c

    def forward(self, features: Tensor, captions: Tensor):
        """Forward pass of RNNDecoder.

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).
            captions (Tensor): Input captions with shape (batch, max_caption_length).

        Returns:
            Tensor: Output predictions with shape (batch, max_caption_length - 1, vocab_size).
        """
        T = captions.shape[1]

        # 词嵌入
        embeddings: Tensor = self.embedding(captions)
        # 初始化隐藏状态
        h, c = self.init_hidden_state(features)

        outputs: list[Tensor] = []
        # Decode step by step
        for t in range(T - 1):
            # Hint 1: apply dropout just before fc
            # Hint 2: use teacher-forcing here
            h, c = self.lstm(embeddings[:, t, :], (h, c))
            output = self.fc(self.dropout(h))
            outputs.append(output)

        # 堆叠输出
        return torch.stack(outputs, dim=1)

    def generate_caption(
        self, features: Tensor, word_map: dict[str, int], max_len=30, beam_size=1
    ):
        """Generate captions with greedy decoding or beam search (during test time).

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).
            word_map (dict[str, int]): Word map from token to index.
            max_len (int): Maximum caption length. Default: 30.
            beam_size (int): Beam size for beam search. Default: 1 (greedy decoding).

        Returns:
            list[list[int]]: Generated captions.
        """
        B = features.shape[0]
        features = features.view(B, -1, features.shape[-1])
        device = features.device

        # 获取特殊标记
        start_token = word_map["<start>"]
        end_token = word_map["<end>"]

        if beam_size == 1:
            # 使用 greedy decoding
            # 初始化隐藏状态
            h, c = self.init_hidden_state(features)

            captions = [[start_token] for _ in range(B)]

            # 逐词生成
            for _ in range(max_len - 1):
                # Hint 1: apply dropout just before fc
                # Hint 2: a caption ends when it reaches the end token
                # 1. 获取当前 predictions
                # 2. 更新 captions
                current_words = torch.tensor(
                    [cap[-1] for cap in captions], dtype=torch.long, device=device
                )
                embeddings = self.embedding(current_words)
                h, c = self.lstm(embeddings, (h, c))
                predictions = self.fc(self.dropout(h)).argmax(dim=1)
                for i, prediction in enumerate(predictions.tolist()):
                    if captions[i][-1] != end_token:
                        captions[i].append(prediction)
                if all(cap[-1] == end_token for cap in captions):
                    break

            return captions
        else:
            # 使用 beam search
            captions: list[list[int]] = []
            # 对每个图像单独进行 beam search
            for i in range(B):
                # 获取单个图像的特征
                img_features = features[i : i + 1]
                # 初始化隐藏状态
                h, c = self.init_hidden_state(img_features)

                # Hint: you may need to store beam as (log_prob, hidden_state, cell_state, tokens)
                # 1. Initialize beam with start token
                # 2. Repeat for max_len - 1 steps:
                # 2.1. Compute log probabilities for all possible next tokens
                #      Note: compute only for non-terminated sequences
                # 2.2. Update beam with top-k (k=beam_size) candidates
                # 2.3. Sort beam by log probability, and keep top-k beams
                # 2.4. Stop if all sequences end with end token
                # 3. Select the sequence with highest log probability
                beam = [(0.0, h, c, [start_token])]
                for _ in range(max_len - 1):
                    candidates = []
                    for score, h_b, c_b, tokens in beam:
                        if tokens[-1] == end_token:
                            candidates.append((score, h_b, c_b, tokens))
                            continue

                        current_word = torch.tensor(
                            [tokens[-1]], dtype=torch.long, device=device
                        )
                        embedding = self.embedding(current_word)
                        h_next, c_next = self.lstm(embedding, (h_b, c_b))
                        logits = self.fc(self.dropout(h_next))
                        log_probs = torch.log_softmax(logits, dim=1).squeeze(0)
                        top_scores, top_words = log_probs.topk(beam_size)

                        for log_prob, word in zip(top_scores, top_words):
                            candidates.append(
                                (
                                    score + log_prob.item(),
                                    h_next,
                                    c_next,
                                    tokens + [word.item()],
                                )
                            )

                    beam = sorted(candidates, key=lambda item: item[0], reverse=True)[
                        :beam_size
                    ]
                    if all(tokens[-1] == end_token for _, _, _, tokens in beam):
                        break

                captions.append(max(beam, key=lambda item: item[0])[3])

            return captions


class RNNDecoderWithAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        vocab_size: int,
        feature_dim: int,
        dropout=0.5,
        rnn_cell=nn.LSTMCell,
        attn_impl=AdditiveAttention,
    ):
        """Initialize RNNDecoder.

        Args:
            embed_dim (int): Embedding dimension.
            hidden_dim (int): Hidden dimension.
            vocab_size (int): Vocabulary size.
            feature_dim (int): Feature dimension.
            dropout (float): Dropout rate. Default: 0.5.
            rnn_cell (Module): RNN cell module. Default: nn.LSTMCell.
            attn_impl (Module): Attention module. Default: AdditiveAttention.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # Decoder expects these layers:
        # - embedding: input embedding
        # - lstm: lstm cell
        # - attention: attention module
        # - init_h: linear projection to init hidden state from image features
        # - init_c: linear projection to init cell state from image features
        # - dropout: output dropout layer
        # - fc: output linear projection
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = rnn_cell(embed_dim + feature_dim, hidden_dim)
        self.attention = attn_impl(feature_dim, hidden_dim)
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Initialize hidden states for RNNDecoder.

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).

        Returns:
            tuple: (hidden_state, cell_state) both with shape (batch, hidden_dim).
        """
        # Same as RNNDecoder, you can copy the code
        mean_features = features.mean(dim=1)
        h = self.init_h(mean_features)
        c = self.init_c(mean_features)
        return h, c

    def forward(self, features: Tensor, captions: Tensor):
        """Forward pass of RNNDecoder.

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).
            captions (Tensor): Input captions with shape (batch, max_caption_length).

        Returns:
            Tensor: Output predictions with shape (batch, max_caption_length - 1, vocab_size).
        """
        T = captions.shape[1]

        # 词嵌入
        embeddings = self.embedding(captions)
        # 初始化隐藏状态
        h, c = self.init_hidden_state(features)

        outputs = []
        # 逐词解码
        for t in range(T - 1):
            # Hint 1: apply dropout just before fc
            # Hint 2: use teacher-forcing here
            context = self.attention(features, h)
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(h))
            outputs.append(output)

        # 堆叠输出
        return torch.stack(outputs, dim=1)

    def generate_caption(
        self, features: Tensor, word_map: dict[str, int], max_len=30, beam_size=1
    ):
        """Generate captions with greedy decoding or beam search (during test time).

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).
            word_map (dict[str, int]): Word map from token to index.
            max_len (int): Maximum caption length. Default: 30.
            beam_size (int): Beam size for beam search. Default: 1 (greedy decoding).

        Returns:
            list[list[int]]: Generated captions.
        """
        B = features.shape[0]
        features = features.view(B, -1, features.shape[-1])
        device = features.device

        # 获取特殊标记
        start_token = word_map["<start>"]
        end_token = word_map["<end>"]

        if beam_size == 1:
            # 使用 greedy decoding
            # 初始化隐藏状态
            h, c = self.init_hidden_state(features)

            captions = [[start_token] for _ in range(B)]

            # 逐词生成
            for _ in range(max_len - 1):
                # Hint 1: apply dropout just before fc
                # Hint 2: a caption ends when it reaches the end token
                # 1. 获取当前 predictions
                # 2. 更新 captions
                current_words = torch.tensor(
                    [cap[-1] for cap in captions], dtype=torch.long, device=device
                )
                embeddings = self.embedding(current_words)
                context = self.attention(features, h)
                lstm_input = torch.cat([embeddings, context], dim=1)
                h, c = self.lstm(lstm_input, (h, c))
                predictions = self.fc(self.dropout(h)).argmax(dim=1)
                for i, prediction in enumerate(predictions.tolist()):
                    if captions[i][-1] != end_token:
                        captions[i].append(prediction)
                if all(cap[-1] == end_token for cap in captions):
                    break

            return captions
        else:
            # 使用 beam search
            captions: list[list[int]] = []
            # 对每个图像单独进行 beam search
            for i in range(B):
                # 获取单个图像的特征
                img_features = features[i : i + 1]
                # 初始化隐藏状态
                h, c = self.init_hidden_state(img_features)

                # Hint: you may need to store beam as (log_prob, hidden_state, cell_state, tokens)
                # 1. Initialize beam with start token
                # 2. Repeat for max_len - 1 steps:
                # 2.1. Compute log probabilities for all possible next tokens
                #      Note: compute only for non-terminated sequences
                # 2.2. Update beam with top-k (k=beam_size) candidates
                # 2.3. Sort beam by log probability, and keep top-k beams
                # 2.4. Stop if all sequences end with end token
                # 3. Select the sequence with highest log probability
                beam = [(0.0, h, c, [start_token])]
                for _ in range(max_len - 1):
                    candidates = []
                    for score, h_b, c_b, tokens in beam:
                        if tokens[-1] == end_token:
                            candidates.append((score, h_b, c_b, tokens))
                            continue

                        current_word = torch.tensor(
                            [tokens[-1]], dtype=torch.long, device=device
                        )
                        embedding = self.embedding(current_word)
                        context = self.attention(img_features, h_b)
                        lstm_input = torch.cat([embedding, context], dim=1)
                        h_next, c_next = self.lstm(lstm_input, (h_b, c_b))
                        logits = self.fc(self.dropout(h_next))
                        log_probs = torch.log_softmax(logits, dim=1).squeeze(0)
                        top_scores, top_words = log_probs.topk(beam_size)

                        for log_prob, word in zip(top_scores, top_words):
                            candidates.append(
                                (
                                    score + log_prob.item(),
                                    h_next,
                                    c_next,
                                    tokens + [word.item()],
                                )
                            )

                    beam = sorted(candidates, key=lambda item: item[0], reverse=True)[
                        :beam_size
                    ]
                    if all(tokens[-1] == end_token for _, _, _, tokens in beam):
                        break

                captions.append(max(beam, key=lambda item: item[0])[3])

            return captions
