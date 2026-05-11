import torch
import torchaudio

def load_mono_and_resample(path:str, tgt_sr:int=-1) -> tuple[torch.Tensor, int]:
    audio, orig_sr = torchaudio.load(path)

    if tgt_sr != -1 and orig_sr != tgt_sr:
        audio = torchaudio.transforms.Resample(orig_sr, tgt_sr)(audio)

    # Convert to mono
    audio = torch.mean(audio, dim=0).unsqueeze(0) # for audio with shape [C, T]

    return audio, orig_sr