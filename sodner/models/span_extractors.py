
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders.embedding import Embedding
from overrides import overrides
import torch
from allennlp.nn import util
from allennlp.modules.time_distributed import TimeDistributed
from torch.nn.init import xavier_normal_
from torch.nn import Conv1d
from math import floor
from allennlp.modules.span_extractors.self_attentive_span_extractor import SelfAttentiveSpanExtractor
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.modules.span_extractors.bidirectional_endpoint_span_extractor import BidirectionalEndpointSpanExtractor

@SpanExtractor.register("pooling")
class PoolingSpanExtractor(SpanExtractor):

    def __init__(self,
                 input_dim: int,
                 combination: str = "max",
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False,
                 use_exclusive_start_indices: bool = False) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._combination = combination
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths
        if bucket_widths:
            raise ConfigurationError("not support")

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            raise ConfigurationError("not support")

        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(num_width_embeddings, span_width_embedding_dim)
        elif not all([num_width_embeddings is None, span_width_embedding_dim is None]):
            raise ConfigurationError("To use a span width embedding representation, you must"
                                     "specify both num_width_buckets and span_width_embedding_dim.")
        else:
            self._span_width_embedding = None

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        combined_dim = self._input_dim
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> None:

        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = util.get_range_vector(max_batch_span_width,
                                                       util.get_device_of(sequence_tensor)).view(1, 1, -1)
        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_mask = (max_span_range_indices <= span_widths).float()
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = util.batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        # Shape: (batch_size, num_spans, embedding_dim)
        # span_embeddings = util.masked_max(span_embeddings, span_mask.unsqueeze(-1), dim=2)
        span_embeddings = util.masked_mean(span_embeddings, span_mask.unsqueeze(-1), dim=2)

        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            span_width_embeddings = self._span_width_embedding(span_widths.squeeze(-1))
            span_embeddings = torch.cat([span_embeddings, span_width_embeddings], -1)

        return span_embeddings


@SpanExtractor.register("conv")
class ConvSpanExtractor(SpanExtractor):

    def __init__(self,
                 input_dim: int,
                 combination: str = "max",
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False,
                 use_exclusive_start_indices: bool = False) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._filter_size = int(combination.split(',')[0])
        if self._filter_size % 2 != 1:
            raise ConfigurationError("The filter size must be an odd.")
        self._combination = combination.split(',')[1]
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths
        if bucket_widths:
            raise ConfigurationError("not support")

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            raise ConfigurationError("not support")

        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(num_width_embeddings, span_width_embedding_dim)
        elif not all([num_width_embeddings is None, span_width_embedding_dim is None]):
            raise ConfigurationError("To use a span width embedding representation, you must"
                                     "specify both num_width_buckets and span_width_embedding_dim.")
        else:
            self._span_width_embedding = None

        self._conv = Conv1d(self._input_dim, self._input_dim, kernel_size=self._filter_size,
                                  padding=int(floor(self._filter_size / 2)))
        xavier_normal_(self._conv.weight)



    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        combined_dim = self._input_dim
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> None:
        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = util.get_range_vector(max_batch_span_width,
                                                       util.get_device_of(sequence_tensor)).view(1, 1, -1)
        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_mask = (max_span_range_indices <= span_widths).float()
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = util.batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        masked_span_embeddings = span_embeddings * span_mask.unsqueeze(-1)

        batch_size, num_spans, max_batch_span_width, embedding_dim = masked_span_embeddings.size()
        # Shape: (batch_size*num_spans, embedding_dim, max_batch_span_width)
        masked_span_embeddings = masked_span_embeddings.view(batch_size * num_spans, max_batch_span_width,
                                                             embedding_dim).transpose(1, 2)

        # Shape: (batch_size, embedding_dim, num_spans*max_batch_span_width)
        conv_span_embeddings = torch.nn.functional.relu(self._conv(masked_span_embeddings))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        conv_span_embeddings = conv_span_embeddings.transpose(1, 2).view(batch_size, num_spans, max_batch_span_width,
                                                                         embedding_dim)

        # Shape: (batch_size, num_spans, embedding_dim)
        span_embeddings = util.masked_max(conv_span_embeddings, span_mask.unsqueeze(-1), dim=2)

        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            span_width_embeddings = self._span_width_embedding(span_widths.squeeze(-1))
            span_embeddings = torch.cat([span_embeddings, span_width_embeddings], -1)

        return span_embeddings


@SpanExtractor.register("attention")
class AttentionSpanExtractor(SpanExtractor):

    def __init__(self,
                 input_dim: int,
                 combination: str = "max",
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False,
                 use_exclusive_start_indices: bool = False) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._combination = combination
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths
        if bucket_widths:
            raise ConfigurationError("not support")

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            raise ConfigurationError("not support")

        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(num_width_embeddings, span_width_embedding_dim)
        elif not all([num_width_embeddings is None, span_width_embedding_dim is None]):
            raise ConfigurationError("To use a span width embedding representation, you must"
                                     "specify both num_width_buckets and span_width_embedding_dim.")
        else:
            self._span_width_embedding = None

        # the allennlp SelfAttentiveSpanExtractor doesn't include span width embedding.
        self._self_attentive = SelfAttentiveSpanExtractor(self._input_dim)


    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        combined_dim = self._input_dim
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> None:

        span_embeddings = self._self_attentive(sequence_tensor, span_indices, sequence_mask, span_indices_mask)

        if self._span_width_embedding is not None:
            # both of shape (batch_size, num_spans, 1)
            span_starts, span_ends = span_indices.split(1, dim=-1)
            # shape (batch_size, num_spans, 1)
            # These span widths are off by 1, because the span ends are `inclusive`.
            span_widths = span_ends - span_starts
            # Embed the span widths and concatenate to the rest of the representations.
            span_width_embeddings = self._span_width_embedding(span_widths.squeeze(-1))
            span_embeddings = torch.cat([span_embeddings, span_width_embeddings], -1)

        return span_embeddings


@SpanExtractor.register("rnn")
class RnnSpanExtractor(SpanExtractor):

    def __init__(self,
                 input_dim: int,
                 combination: str = "x,y",
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False,
                 use_exclusive_start_indices: bool = False) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._combination = combination

        self._encoder = PytorchSeq2SeqWrapper(StackedBidirectionalLstm(self._input_dim, int(floor(self._input_dim / 2)), 1))
        self._span_extractor = BidirectionalEndpointSpanExtractor(self._input_dim, "y", "y",
                                                                  num_width_embeddings, span_width_embedding_dim, bucket_widths)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._span_extractor.get_output_dim()

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> None:

        # Shape: (batch_size, sequence_length, embedding_dim)
        encoder_sequence_tensor = self._encoder(sequence_tensor, sequence_mask)

        # Shape: (batch_size, num_spans, embedding_dim)
        span_embeddings = self._span_extractor(encoder_sequence_tensor, span_indices, sequence_mask, span_indices_mask)

        return span_embeddings
