"""Small fully convolutional line recognizer, with CTC blank at class zero."""
import torch
from torch import nn


class LineCNN(nn.Module):
    def __init__(self, classes: int, channels: int = 32):
        super().__init__()
        blocks = []
        source = 1
        for dest, stride in ((channels, (2, 2)), (channels * 2, (2, 2)),
                             (channels * 4, (2, 2)), (channels * 4, (2, 1))):
            blocks.extend([nn.Conv2d(source, dest, 3, stride=stride, padding=1), nn.ReLU()])
            source = dest
        self.features = nn.Sequential(*blocks)
        self.context = nn.Sequential(nn.Conv1d(source, source, 5, padding=2), nn.ReLU())
        self.classifier = nn.Linear(source, classes)

    @staticmethod
    def output_lengths(widths):
        # Three padded width-stride-two convolutions: ceil(width / 8).
        return (widths + 7) // 8

    def forward(self, images):
        features = self.features(1 - images).mean(dim=2)
        features = self.context(features).permute(2, 0, 1)
        return self.classifier(features)  # [T, B, C]; log-softmax is done in float32.


def ctc_loss(logits, batch):
    lengths = LineCNN.output_lengths(batch.image_widths)
    minimum = torch.tensor([
        len(text) + sum(a == b for a, b in zip(text, text[1:])) for text in batch.texts
    ], dtype=torch.int64)
    if torch.any(lengths < minimum) or torch.any(lengths > logits.shape[0]):
        raise ValueError('CNN output is too short for CTC targets, including repeated characters')
    return nn.functional.ctc_loss(
        logits.float().log_softmax(2), batch.targets.to(logits.device),
        lengths, batch.target_lengths, blank=0, zero_infinity=False,
    )
